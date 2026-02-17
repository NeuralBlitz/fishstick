"""
Causal Inference and Structural Causal Models.

Implements:
- Do-calculus for interventions
- Counterfactual reasoning
- Causal discovery (PC algorithm, NOTEARS)
- Instrumental variables
- Causal effect estimation
"""

from typing import Optional, Tuple, Dict, List, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class CausalGraph:
    """Represents a causal DAG."""

    n_nodes: int
    adjacency: np.ndarray  # Binary adjacency matrix
    node_names: Optional[List[str]] = None

    def __post_init__(self):
        if self.node_names is None:
            self.node_names = [f"X{i}" for i in range(self.n_nodes)]

    def parents(self, node: int) -> List[int]:
        """Get parents of a node."""
        return list(np.where(self.adjacency[:, node] == 1)[0])

    def children(self, node: int) -> List[int]:
        """Get children of a node."""
        return list(np.where(self.adjacency[node, :] == 1)[0])

    def is_dag(self) -> bool:
        """Check if graph is a DAG."""
        # Use topological sort
        visited = [False] * self.n_nodes
        rec_stack = [False] * self.n_nodes

        def has_cycle(v):
            visited[v] = True
            rec_stack[v] = True

            for child in self.children(v):
                if not visited[child] and has_cycle(child):
                    return True
                elif rec_stack[child]:
                    return True

            rec_stack[v] = False
            return False

        for node in range(self.n_nodes):
            if not visited[node]:
                if has_cycle(node):
                    return False
        return True


class StructuralEquation(nn.Module):
    """
    Neural structural equation: X_i = f_i(PA_i, ε_i).

    Represents one node in a structural causal model.
    """

    def __init__(
        self,
        n_parents: int,
        noise_dim: int = 1,
        hidden_dim: int = 64,
        nonlinearity: str = "mlp",
    ):
        super().__init__()
        self.n_parents = n_parents
        self.noise_dim = noise_dim

        input_dim = n_parents + noise_dim

        if nonlinearity == "mlp":
            self.f = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif nonlinearity == "additive":
            self.parent_weight = nn.Parameter(torch.randn(n_parents))
            self.noise_weight = nn.Parameter(torch.tensor(1.0))
            self.f = lambda parents, noise: (
                parents @ self.parent_weight + self.noise_weight * noise
            )
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

        self.nonlinearity = nonlinearity

    def forward(
        self,
        parents: Optional[Tensor],
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute structural equation.

        Args:
            parents: Parent values [batch, n_parents]
            noise: Noise values [batch, noise_dim]

        Returns:
            Node value [batch, 1]
        """
        if noise is None:
            noise = torch.randn(parents.size(0), self.noise_dim, device=parents.device)

        if self.nonlinearity == "additive":
            return self.f(parents, noise).unsqueeze(-1)
        else:
            inputs = (
                torch.cat([parents, noise], dim=-1) if parents is not None else noise
            )
            return self.f(inputs)


class StructuralCausalModel(nn.Module):
    """
    Complete structural causal model (SCM).

    A collection of structural equations organized as a DAG.
    """

    def __init__(
        self,
        graph: CausalGraph,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.graph = graph

        # Create structural equation for each node
        self.equations = nn.ModuleList()
        for node in range(graph.n_nodes):
            n_parents = len(graph.parents(node))
            self.equations.append(StructuralEquation(n_parents, hidden_dim=hidden_dim))

    def forward(
        self,
        interventions: Optional[Dict[int, Tensor]] = None,
    ) -> Tensor:
        """
        Sample from the SCM (observational or interventional).

        Args:
            interventions: Dict mapping node indices to intervention values

        Returns:
            Sampled values [batch, n_nodes]
        """
        if interventions is None:
            interventions = {}

        batch_size = next(iter(interventions.values())).size(0) if interventions else 1
        device = next(iter(interventions.values())).device if interventions else "cpu"

        values = {}

        # Topological order
        topo_order = self._topological_sort()

        for node in topo_order:
            if node in interventions:
                values[node] = interventions[node]
            else:
                parents = self.graph.parents(node)
                if parents:
                    parent_values = torch.cat([values[p] for p in parents], dim=-1)
                else:
                    parent_values = torch.zeros(batch_size, 0, device=device)

                node_value = self.equations[node](parent_values)
                values[node] = node_value

        # Stack in order
        result = torch.cat([values[i] for i in range(self.graph.n_nodes)], dim=-1)
        return result

    def do_calculus(self, intervention_node: int, value: Tensor) -> Tensor:
        """
        Perform do-operator: P(Y | do(X=x)).

        Args:
            intervention_node: Node to intervene on
            value: Intervention value

        Returns:
            Distribution of other nodes after intervention
        """
        interventions = {intervention_node: value}
        return self.forward(interventions)

    def counterfactual(
        self,
        evidence: Dict[int, Tensor],
        intervention: Tuple[int, Tensor],
    ) -> Tensor:
        """
        Counterfactual inference: "What if X had been x'?"

        Three steps:
        1. Abduction: Infer noise from evidence
        2. Action: Apply intervention
        3. Prediction: Forward pass with inferred noise

        Args:
            evidence: Observed evidence
            intervention: (node, value) to intervene on

        Returns:
            Counterfactual outcome
        """
        # Step 1: Abduction (simplified - assumes additive noise)
        # In practice, this requires inference over the noise

        # Step 2 & 3: Apply intervention and predict
        intervention_node, intervention_value = intervention
        interventions = {**evidence, intervention_node: intervention_value}

        return self.forward(interventions)

    def _topological_sort(self) -> List[int]:
        """Topological sort of the DAG."""
        in_degree = [0] * self.graph.n_nodes
        for node in range(self.graph.n_nodes):
            for child in self.graph.children(node):
                in_degree[child] += 1

        queue = [i for i in range(self.graph.n_nodes) if in_degree[i] == 0]
        topo_order = []

        while queue:
            node = queue.pop(0)
            topo_order.append(node)

            for child in self.graph.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return topo_order


class CausalDiscovery:
    """
    Algorithms for discovering causal structure from data.
    """

    @staticmethod
    def pc_algorithm(
        data: np.ndarray,
        alpha: float = 0.05,
        independence_test: str = "fisher",
    ) -> np.ndarray:
        """
        PC algorithm for causal discovery.

        Args:
            data: [n_samples, n_features] observational data
            alpha: Significance level for independence tests
            independence_test: Type of independence test

        Returns:
            Adjacency matrix of learned DAG
        """
        n_samples, n_features = data.shape

        # Initialize fully connected graph
        adjacency = np.ones((n_features, n_features)) - np.eye(n_features)

        # Step 1: Skeleton identification (simplified)
        # In practice, use conditional independence tests
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Simple correlation test
                corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                if abs(corr) < alpha:
                    adjacency[i, j] = 0
                    adjacency[j, i] = 0

        # Step 2: Orientation (simplified)
        # Use v-structures and propagation

        return adjacency

    @staticmethod
    def notears(
        data: np.ndarray,
        lambda1: float = 0.1,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
    ) -> np.ndarray:
        """
        NOTEARS: Gradient-based DAG learning.

        From: DAGs with NO TEARS (Zheng et al., 2018)

        Args:
            data: [n_samples, n_features] observational data
            lambda1: L1 regularization
            max_iter: Maximum iterations

        Returns:
            Weighted adjacency matrix
        """
        n_samples, n_features = data.shape

        # Normalize data
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

        # Initialize
        W = np.zeros((n_features, n_features))
        rho = 1.0
        alpha = 0.0
        h = np.inf

        for iteration in range(max_iter):
            # Solve with augmented Lagrangian
            W_new = CausalDiscovery._notears_solver(data, W, rho, alpha, lambda1)

            # Check acyclicity constraint
            h_new = CausalDiscovery._h(W_new)

            if h_new > 0.25 * h:
                rho *= 10

            W = W_new
            alpha += rho * h_new
            h = h_new

            if h <= h_tol or rho >= rho_max:
                break

        # Threshold to get adjacency
        adjacency = (np.abs(W) > 0.3).astype(int)
        np.fill_diagonal(adjacency, 0)

        return adjacency

    @staticmethod
    def _notears_solver(
        data: np.ndarray,
        W: np.ndarray,
        rho: float,
        alpha: float,
        lambda1: float,
    ) -> np.ndarray:
        """Inner solver for NOTEARS."""
        n_samples, n_features = data.shape

        # Gradient descent (simplified)
        W_new = W.copy()
        lr = 0.001

        for _ in range(100):
            # Least squares gradient
            grad_ls = (2.0 / n_samples) * data.T @ (data @ W_new - data)

            # Acyclicity gradient
            grad_h = CausalDiscovery._grad_h(W_new)

            # L1 gradient
            grad_l1 = lambda1 * np.sign(W_new)

            # Total gradient
            grad = (
                grad_ls + grad_h * (rho * CausalDiscovery._h(W_new) + alpha) + grad_l1
            )

            W_new -= lr * grad

        return W_new

    @staticmethod
    def _h(W: np.ndarray) -> float:
        """Acyclicity constraint: h(W) = tr(e^(W⊙W)) - d = 0."""
        d = W.shape[0]
        M = np.eye(d) + W * W / d
        E = np.linalg.matrix_power(M, d)
        h = np.trace(E) - d
        return h

    @staticmethod
    def _grad_h(W: np.ndarray) -> np.ndarray:
        """Gradient of acyclicity constraint."""
        d = W.shape[0]
        M = np.eye(d) + W * W / d
        E = np.linalg.matrix_power(M, d - 1)
        grad = E.T * 2 * W / d
        return grad


class InstrumentalVariableEstimator:
    """
    Estimate causal effects using instrumental variables.
    """

    def __init__(self, method: str = "2sls"):
        self.method = method
        self.stage1_model = None
        self.stage2_model = None

    def fit(
        self,
        Z: Tensor,  # Instrument [n_samples]
        X: Tensor,  # Treatment [n_samples]
        Y: Tensor,  # Outcome [n_samples]
    ) -> float:
        """
        Estimate causal effect using instrumental variable.

        Returns:
            Estimated causal effect
        """
        if self.method == "2sls":
            # Two-stage least squares
            # Stage 1: X ~ Z
            self.stage1_model = self._fit_linear(Z, X)
            X_hat = self._predict_linear(Z, self.stage1_model)

            # Stage 2: Y ~ X_hat
            self.stage2_model = self._fit_linear(X_hat, Y)
            effect = self.stage2_model[0].item()

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return effect

    def _fit_linear(self, X: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:
        """Fit linear regression: Y = aX + b."""
        X_with_bias = torch.stack([X, torch.ones_like(X)], dim=-1)

        # Least squares: (X^T X)^-1 X^T Y
        XtX = X_with_bias.T @ X_with_bias
        XtY = X_with_bias.T @ Y

        params = torch.linalg.solve(XtX, XtY)
        return params[0], params[1]  # slope, intercept

    def _predict_linear(self, X: Tensor, model: Tuple[Tensor, Tensor]) -> Tensor:
        """Predict using linear model."""
        slope, intercept = model
        return slope * X + intercept


class PropensityScoreMatching:
    """
    Estimate causal effects using propensity score matching.
    """

    def __init__(self, propensity_model: Optional[nn.Module] = None):
        if propensity_model is None:
            self.propensity_model = nn.Sequential(
                nn.Linear(10, 64),  # Assuming 10 covariates
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        else:
            self.propensity_model = propensity_model

    def fit_propensity(self, covariates: Tensor, treatment: Tensor) -> None:
        """Fit propensity score model."""
        optimizer = torch.optim.Adam(self.propensity_model.parameters())

        for _ in range(1000):
            optimizer.zero_grad()

            propensity = self.propensity_model(covariates).squeeze()
            loss = F.binary_cross_entropy(propensity, treatment)

            loss.backward()
            optimizer.step()

    def estimate_ate(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
    ) -> float:
        """
        Estimate Average Treatment Effect (ATE).

        ATE = E[Y(1) - Y(0)]
        """
        # Get propensity scores
        with torch.no_grad():
            propensity = self.propensity_model(covariates).squeeze()

        # Inverse probability weighting
        treated = treatment == 1
        untreated = treatment == 0

        # IPW estimator
        weights_treated = 1.0 / propensity[treated]
        weights_untreated = 1.0 / (1 - propensity[untreated])

        ate = (weights_treated * outcome[treated]).sum() / weights_treated.sum() - (
            weights_untreated * outcome[untreated]
        ).sum() / weights_untreated.sum()

        return ate.item()


class DoublyRobustEstimator:
    """
    Doubly robust estimator combining propensity score and outcome modeling.
    """

    def __init__(
        self,
        propensity_model: nn.Module,
        outcome_model: nn.Module,
    ):
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model

    def estimate_ate(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
    ) -> float:
        """
        Estimate ATE using doubly robust method.

        This estimator is consistent if either the propensity model
        OR the outcome model is correctly specified.
        """
        # Get propensity scores
        with torch.no_grad():
            propensity = self.propensity_model(covariates).squeeze()

            # Predict counterfactual outcomes
            cov_treated = torch.cat(
                [covariates, torch.ones_like(treatment.unsqueeze(-1))], dim=-1
            )
            cov_untreated = torch.cat(
                [covariates, torch.zeros_like(treatment.unsqueeze(-1))], dim=-1
            )

            mu1 = self.outcome_model(cov_treated).squeeze()
            mu0 = self.outcome_model(cov_untreated).squeeze()

        # Doubly robust estimator
        n = len(outcome)

        term1 = treatment * (outcome - mu1) / propensity + mu1

        term0 = (1 - treatment) * (outcome - mu0) / (1 - propensity) + mu0

        ate = (term1 - term0).mean()

        return ate.item()


class CausalVAE(nn.Module):
    """
    Variational autoencoder with causal structure.

    Disentangles causal and non-causal latent factors.
    """

    def __init__(
        self,
        input_dim: int,
        causal_dim: int,
        noise_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.causal_dim = causal_dim
        self.noise_dim = noise_dim
        latent_dim = causal_dim + noise_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Causal decoder (structured)
        self.causal_decoder = nn.Sequential(
            nn.Linear(causal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, causal_dim),
        )

        # Full decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode to latent distribution."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode from latent."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        x_recon = self.decode(z)

        # Split causal and non-causal
        z_causal = z[:, : self.causal_dim]
        z_noise = z[:, self.causal_dim :]

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z_causal": z_causal,
            "z_noise": z_noise,
        }

    def intervene(self, x: Tensor, intervention: Dict[int, float]) -> Tensor:
        """Generate counterfactual by intervening on causal latents."""
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)

            # Apply intervention
            for idx, value in intervention.items():
                z[:, idx] = value

            x_cf = self.decode(z)

        return x_cf
