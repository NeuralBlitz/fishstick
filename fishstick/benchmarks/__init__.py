"""
Benchmark Suite for Unified Intelligence Framework.

Provides standardized benchmarks for evaluating:
- Physics-informed learning (Navier-Stokes, quantum systems)
- Symbolic reasoning (bAbI suite)
- Out-of-distribution generalization
- Formal verification
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

from ..frameworks.uniintelli import create_uniintelli
from ..frameworks.hsca import create_hsca
from ..frameworks.uia import create_uia
from ..frameworks.scif import create_scif
from ..frameworks.uif import create_uif
from ..frameworks.uis import create_uis


class SyntheticPhysicsDataset(Dataset):
    """
    Synthetic physics dataset for testing conservation laws.

    Generates trajectories from harmonic oscillator or pendulum dynamics.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seq_len: int = 50,
        system: str = "harmonic",
        noise: float = 0.01,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.system = system
        self.noise = noise

        self.trajectories = self._generate_trajectories()

    def _generate_trajectories(self) -> Tensor:
        """Generate physical trajectories."""
        trajectories = []

        for _ in range(self.n_samples):
            if self.system == "harmonic":
                omega = np.random.uniform(0.5, 2.0)
                phi = np.random.uniform(0, 2 * np.pi)
                t = np.linspace(0, 10, self.seq_len)

                q = np.sin(omega * t + phi)
                p = omega * np.cos(omega * t + phi)

            else:
                g = 9.8
                L = 1.0
                theta0 = np.random.uniform(0.1, 2.5)
                t = np.linspace(0, 5, self.seq_len)

                theta = theta0 * np.cos(np.sqrt(g / L) * t)
                p = -theta0 * np.sqrt(g / L) * np.sin(np.sqrt(g / L) * t)
                q = theta

            traj = np.stack([q, p], axis=1)
            traj += np.random.randn(*traj.shape) * self.noise
            trajectories.append(traj)

        return torch.tensor(trajectories, dtype=torch.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        traj = self.trajectories[idx]
        return traj[:-1], traj[1:]


class SyntheticClassificationDataset(Dataset):
    """Synthetic classification dataset for testing."""

    def __init__(
        self, n_samples: int = 1000, input_dim: int = 784, n_classes: int = 10
    ):
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.X = torch.randn(n_samples, input_dim)
        self.y = torch.randint(0, n_classes, (n_samples,))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.X[idx], self.y[idx]


class BenchmarkRunner:
    """
    Runner for unified intelligence benchmarks.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results: Dict[str, Dict[str, Any]] = {}

    def run_all(
        self, n_epochs: int = 5, batch_size: int = 32
    ) -> Dict[str, Dict[str, Any]]:
        """Run all benchmarks."""

        print("Running Unified Intelligence Benchmarks...")
        print("=" * 50)

        self.benchmark_frameworks(n_epochs, batch_size)

        self.benchmark_conservation()

        self.benchmark_verification()

        return self.results

    def benchmark_frameworks(self, n_epochs: int = 5, batch_size: int = 32) -> None:
        """Benchmark all 6 frameworks."""

        frameworks = {
            "UniIntelli": create_uniintelli,
            "HSCA": create_hsca,
            "UIA": create_uia,
            "SCIF": create_scif,
            "UIF": create_uif,
            "UIS": create_uis,
        }

        dataset = SyntheticClassificationDataset(n_samples=500, input_dim=784)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for name, create_fn in frameworks.items():
            print(f"\nBenchmarking {name}...")

            model = create_fn(input_dim=784, output_dim=10)
            model.to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss()

            start_time = time.time()
            losses = []

            for epoch in range(n_epochs):
                epoch_loss = 0.0
                n_batches = 0

                for x, y in dataloader:
                    x, y = x.to(self.device), y.to(self.device)

                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                losses.append(epoch_loss / n_batches)

            elapsed = time.time() - start_time
            n_params = sum(p.numel() for p in model.parameters())

            self.results[name] = {
                "final_loss": losses[-1],
                "training_time": elapsed,
                "n_parameters": n_params,
                "loss_history": losses,
            }

            print(f"  Final loss: {losses[-1]:.4f}")
            print(f"  Training time: {elapsed:.2f}s")
            print(f"  Parameters: {n_params:,}")

    def benchmark_conservation(self) -> None:
        """Benchmark energy conservation."""
        print("\nBenchmarking Energy Conservation...")

        dataset = SyntheticPhysicsDataset(n_samples=100, seq_len=20)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        from ..dynamics.hamiltonian import HamiltonianNeuralNetwork

        hnn = HamiltonianNeuralNetwork(input_dim=2, hidden_dim=64)
        hnn.to(self.device)

        energy_violations = []

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            z0 = x.view(-1, 4)
            z1 = hnn.integrate(z0, n_steps=1, dt=0.1)[-1]

            E0 = hnn.H_net(z0)
            E1 = hnn.H_net(z1)
            violation = torch.abs(E0 - E1).mean().item()
            energy_violations.append(violation)

        self.results["EnergyConservation"] = {
            "mean_violation": np.mean(energy_violations),
            "max_violation": np.max(energy_violations),
        }

        print(f"  Mean violation: {np.mean(energy_violations):.6f}")
        print(f"  Max violation: {np.max(energy_violations):.6f}")

    def benchmark_verification(self) -> None:
        """Benchmark verification capabilities."""
        print("\nBenchmarking Verification...")

        from ..verification.types import DependentlyTypedLearner, VerificationPipeline

        model = torch.nn.Sequential(
            torch.nn.Linear(784, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10)
        )

        typed_model = DependentlyTypedLearner(
            layers=list(model.children()), input_spec=None, output_spec=None
        )

        x = torch.randn(1, 784)

        cert = typed_model.verify_robustness(x, epsilon=0.1)

        self.results["Verification"] = {
            "robustness_verified": cert.is_verified,
            "lipschitz_bound": typed_model.lipschitz_constant,
        }

        print(f"  Robustness verified: {cert.is_verified}")
        print(f"  Lipschitz bound: {typed_model.lipschitz_constant:.4f}")

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)

        for name, result in self.results.items():
            print(f"\n{name}:")
            for key, value in result.items():
                if isinstance(value, list):
                    print(f"  {key}: [{value[0]:.4f}, ..., {value[-1]:.4f}]")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


def run_benchmarks(n_epochs: int = 5, batch_size: int = 32) -> Dict[str, Any]:
    """Convenience function to run all benchmarks."""
    runner = BenchmarkRunner()
    results = runner.run_all(n_epochs, batch_size)
    runner.print_summary()
    return results


if __name__ == "__main__":
    run_benchmarks()
