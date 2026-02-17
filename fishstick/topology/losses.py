"""
TDA-based Loss Functions.

Provides loss functions for geometric deep learning that incorporate
topological priors and persistence-based regularization.
"""

from typing import List, Optional, Callable, Dict, Tuple
import torch
from torch import Tensor, nn
import numpy as np

from .persistence import (
    PersistentHomology,
    PersistenceDiagram,
    BirthDeathPair,
    wasserstein_distance,
    bottleneck_distance,
)
from .vietoris_rips import VietorisRipsComplex
from .features import PersistentEntropy, Silhouette


class PersistentHomologyLoss(nn.Module):
    """
    Persistent Homology Loss.

    Encourages the model to learn representations with specific
    topological properties using persistent homology.
    """

    def __init__(
        self,
        target_dimensions: List[int] = [0, 1],
        weight: float = 1.0,
        max_dimension: int = 2,
    ):
        """
        Initialize persistent homology loss.

        Args:
            target_dimensions: Dimensions to compute persistence for
            weight: Loss weight
            max_dimension: Maximum homology dimension
        """
        super().__init__()
        self.target_dimensions = target_dimensions
        self.weight = weight
        self.max_dimension = max_dimension
        self.ph = PersistentHomology(max_dimension=max_dimension)

    def forward(
        self,
        representations: Tensor,
        target_topology: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute persistence loss from representations.

        Args:
            representations: Input representations [batch, features]
            target_topology: Optional target persistence values

        Returns:
            Loss value
        """
        batch_size = representations.shape[0]
        loss = 0.0

        for i in range(batch_size):
            rep = representations[i : i + 1]

            diagrams = self.ph.compute_from_distance(rep)

            for dim in self.target_dimensions:
                if dim < len(diagrams):
                    diagram = diagrams[dim]

                    persistence = torch.tensor(
                        [p.persistence for p in diagram.pairs], dtype=torch.float32
                    )

                    if len(persistence) > 0:
                        loss += torch.mean(persistence)

        loss = loss / (batch_size * len(self.target_dimensions))

        return self.weight * loss


class DiagramDistanceLoss(nn.Module):
    """
    Diagram Distance Loss.

    Uses Wasserstein or bottleneck distance between persistence
    diagrams as a loss for learning topological representations.
    """

    def __init__(
        self,
        distance_type: str = "wasserstein",
        p: float = 2.0,
        weight: float = 1.0,
    ):
        """
        Initialize diagram distance loss.

        Args:
            distance_type: Type of distance ('wasserstein', 'bottleneck')
            p: Order for Wasserstein distance
            weight: Loss weight
        """
        super().__init__()
        self.distance_type = distance_type
        self.p = p
        self.weight = weight

    def forward(
        self,
        representations1: Tensor,
        representations2: Tensor,
    ) -> Tensor:
        """
        Compute diagram distance loss between two representations.

        Args:
            representations1: First set of representations
            representations2: Second set of representations

        Returns:
            Distance loss value
        """
        ph = PersistentHomology()

        diagrams1 = ph.compute_from_distance(representations1)
        diagrams2 = ph.compute_from_distance(representations2)

        total_distance = 0.0

        for d1, d2 in zip(diagrams1, diagrams2):
            if self.distance_type == "wasserstein":
                dist = wasserstein_distance(d1, d2, p=self.p)
            else:
                dist = bottleneck_distance(d1, d2)

            total_distance += dist

        return self.weight * total_distance


class PersistentEntropyLoss(nn.Module):
    """
    Persistent Entropy Loss.

    Uses persistent entropy as a regularization term to encourage
    specific topological complexity in learned representations.
    """

    def __init__(
        self,
        target_entropy: Optional[float] = None,
        weight: float = 1.0,
        dimensions: List[int] = [0, 1],
    ):
        """
        Initialize persistent entropy loss.

        Args:
            target_entropy: Target entropy value (None = maximize)
            weight: Loss weight
            dimensions: Dimensions to compute entropy for
        """
        super().__init__()
        self.target_entropy = target_entropy
        self.weight = weight
        self.dimensions = dimensions
        self.entropy_computer = PersistentEntropy()

    def forward(self, representations: Tensor) -> Tensor:
        """
        Compute entropy loss.

        Args:
            representations: Input representations

        Returns:
            Entropy loss value
        """
        ph = PersistentHomology()
        diagrams = ph.compute_from_distance(representations)

        loss = 0.0

        for dim in self.dimensions:
            if dim < len(diagrams):
                diagram = diagrams[dim]
                entropy = self.entropy_computer.compute(diagram)

                if self.target_entropy is not None:
                    loss += (entropy - self.target_entropy) ** 2
                else:
                    loss -= entropy

        return self.weight * loss


class TopologicalRegularization(nn.Module):
    """
    Topological Regularization.

    Provides various regularization terms based on topological
    features of neural network activations.
    """

    def __init__(
        self,
        regularization_type: str = "persistence",
        weight: float = 0.01,
        dimensions: List[int] = [0],
    ):
        """
        Initialize topological regularization.

        Args:
            regularization_type: Type of regularization
            weight: Regularization weight
            dimensions: Dimensions to regularize
        """
        super().__init__()
        self.regularization_type = regularization_type
        self.weight = weight
        self.dimensions = dimensions
        self.silhouette = Silhouette()

    def forward(self, representations: Tensor) -> Tensor:
        """
        Compute topological regularization.

        Args:
            representations: Input representations [batch, features]

        Returns:
            Regularization loss
        """
        ph = PersistentHomology()
        diagrams = ph.compute_from_distance(representations)

        loss = 0.0

        for dim in self.dimensions:
            if dim < len(diagrams):
                diagram = diagrams[dim]

                if self.regularization_type == "persistence":
                    persistences = diagram.persistences
                    if len(persistences) > 0:
                        loss += torch.mean(persistences)

                elif self.regularization_type == "silhouette":
                    sil = self.silhouette.compute(diagram)
                    loss += sil

                elif self.regularization_type == "entropy":
                    entropy = PersistentEntropy()
                    loss -= entropy.compute(diagram)

                elif self.regularization_type == "stability":
                    persistences = diagram.persistences
                    if len(persistences) > 0:
                        stability = torch.var(persistences)
                        loss += stability

        return self.weight * loss


class TopologicalContrastiveLoss(nn.Module):
    """
    Topological Contrastive Loss.

    Contrastive loss that considers topological similarity
    between positive pairs.
    """

    def __init__(
        self,
        margin: float = 1.0,
        topological_weight: float = 0.5,
    ):
        super().__init__()
        self.margin = margin
        self.topological_weight = topological_weight
        self.ph = PersistentHomology()

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        """
        Compute topological contrastive loss.

        Args:
            anchor: Anchor representations
            positive: Positive pairs
            negative: Negative pairs

        Returns:
            Contrastive loss
        """
        pos_dist = self._compute_topological_distance(anchor, positive)
        neg_dist = self._compute_topological_distance(anchor, negative)

        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)

        return loss

    def _compute_topological_distance(
        self,
        representations1: Tensor,
        representations2: Tensor,
    ) -> Tensor:
        """Compute topological distance between representations."""
        diagrams1 = self.ph.compute_from_distance(representations1)
        diagrams2 = self.ph.compute_from_distance(representations2)

        dist = 0.0

        for d1, d2 in zip(diagrams1, diagrams2):
            dist += wasserstein_distance(d1, d2)

        return dist


class MapperBasedLoss(nn.Module):
    """
    Mapper-based Loss.

    Uses Mapper algorithm output as a loss signal for
    learning representations with desired topological structure.
    """

    def __init__(
        self,
        n_cubes: int = 5,
        target_components: int = 1,
        weight: float = 1.0,
    ):
        super().__init__()
        self.n_cubes = n_cubes
        self.target_components = target_components
        self.weight = weight

    def forward(
        self,
        representations: Tensor,
        filter_fn: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute Mapper-based loss.

        Args:
            representations: Input representations
            filter_fn: Optional filter function values

        Returns:
            Loss value based on connected components
        """
        from .mapper import Mapper

        mapper = Mapper(n_cubes=self.n_cubes)

        clusters, edges = mapper.fit_transform(representations, filter_fn)

        from .mapper import MapperGraph

        graph = MapperGraph(clusters, edges)

        components = graph.get_connected_components()
        n_components = len(components)

        loss = abs(n_components - self.target_components)

        return self.weight * torch.tensor(loss, dtype=torch.float32)


class PersistenceNLLLoss(nn.Module):
    """
    Persistence Negative Log-Likelihood Loss.

    Converts persistence diagram into a probability distribution
    and computes negative log-likelihood.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(
        self,
        representations: Tensor,
        target_distribution: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute persistence NLL loss.

        Args:
            representations: Input representations
            target_distribution: Optional target distribution

        Returns:
            NLL loss
        """
        ph = PersistentHomology()
        diagrams = ph.compute_from_distance(representations)

        if len(diagrams) == 0:
            return torch.tensor(0.0)

        persistences = diagrams[0].persistences

        if len(persistences) == 0:
            return torch.tensor(0.0)

        weights = persistences / (torch.sum(persistences) + 1e-10)

        weights = weights / self.temperature

        weights = torch.softmax(weights, dim=0)

        if target_distribution is not None:
            nll = -torch.sum(target_distribution * torch.log(weights + 1e-10))
        else:
            nll = -torch.mean(torch.log(weights + 1e-10))

        return self.weight * nll


class TopologicalVAELoss(nn.Module):
    """
    Topological VAE Loss.

    Combines VAE reconstruction and KL terms with topological
    regularization for learning topologically meaningful latent spaces.
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 0.1,
        topological_weight: float = 0.5,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.topological_weight = topological_weight
        self.regularizer = TopologicalRegularization(
            regularization_type="silhouette",
            weight=1.0,
        )

    def forward(
        self,
        recon_x: Tensor,
        x: Tensor,
        mu: Tensor,
        logvar: Tensor,
        representations: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute topological VAE loss.

        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Latent mean
            logvar: Latent log variance
            representations: Latent representations

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        topo_loss = self.regularizer(representations)

        total_loss = (
            self.reconstruction_weight * recon_loss
            + self.kl_weight * kl_loss
            + self.topological_weight * topo_loss
        )

        loss_dict = {
            "reconstruction": recon_loss,
            "kl": kl_loss,
            "topological": topo_loss,
            "total": total_loss,
        }

        return total_loss, loss_dict


class GraphTopologicalLoss(nn.Module):
    """
    Graph Topological Loss.

    Computes topological features of graph structures and uses
    them as loss for graph neural networks.
    """

    def __init__(
        self,
        weight: float = 1.0,
        use_homology: bool = True,
    ):
        super().__init__()
        self.weight = weight
        self.use_homology = use_homology
        self.ph = PersistentHomology(max_dimension=1)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Compute graph topological loss.

        Args:
            node_features: Node feature matrix
            edge_index: Edge index tensor [2, n_edges]

        Returns:
            Topological loss
        """
        num_nodes = node_features.shape[0]
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1

        eigenvalues = torch.linalg.eigvalsh(adj)

        spectrum_loss = torch.var(eigenvalues)

        if self.use_homology:
            edge_list = edge_index.t().tolist()
            points = node_features

            distances = torch.cdist(points, points)

            vr_complex = VietorisRipsComplex(max_dimension=1)
            simplices, filtrations = vr_complex.build_from_points(points)

            boundary_matrices = self.ph._build_boundary_matrices(simplices)
            diagrams = self.ph.compute(filtrations, boundary_matrices)

            if len(diagrams) > 1:
                h1_diagram = diagrams[1]
                persistences = h1_diagram.persistences
                if len(persistences) > 0:
                    spectrum_loss = spectrum_loss + torch.mean(persistences)

        return self.weight * spectrum_loss
