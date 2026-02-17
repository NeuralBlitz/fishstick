"""
Deployment Module for fishstick

Production deployment utilities:
- Docker containerization
- Cloud deployment (AWS, GCP, Azure helpers)
- Model registry and versioning
- A/B testing support
- Monitoring and health checks
"""

from typing import Optional, Dict, Any, List
import os
import json
from pathlib import Path
from datetime import datetime


class DockerBuilder:
    """
    Build Docker containers for fishstick models.
    """

    def __init__(self, base_image: str = "python:3.11-slim"):
        self.base_image = base_image

    def generate_dockerfile(
        self,
        model_path: str,
        port: int = 8000,
        requirements: Optional[List[str]] = None,
    ) -> str:
        """
        Generate Dockerfile for model serving.

        Args:
            model_path: Path to model file
            port: Port to expose
            requirements: Additional pip requirements

        Returns:
            Dockerfile content
        """
        dockerfile = f"""FROM {self.base_image}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY {model_path} ./model.pt
COPY . .

# Expose port
EXPOSE {port}

# Run API server
CMD ["python", "-m", "fishstick.cli", "serve", "--model", "model.pt", "--port", "{port}"]
"""
        return dockerfile

    def build(
        self, tag: str, model_path: str, context_dir: str = ".", push: bool = False
    ) -> str:
        """
        Build Docker image.

        Args:
            tag: Image tag (e.g., "fishstick-model:v1")
            model_path: Path to model
            context_dir: Build context directory
            push: Whether to push to registry

        Returns:
            Image tag
        """
        import subprocess

        # Generate Dockerfile
        dockerfile_content = self.generate_dockerfile(model_path)
        dockerfile_path = Path(context_dir) / "Dockerfile"

        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        # Build image
        cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile_path), context_dir]
        subprocess.run(cmd, check=True)

        print(f"✓ Built Docker image: {tag}")

        # Push if requested
        if push:
            subprocess.run(["docker", "push", tag], check=True)
            print(f"✓ Pushed to registry: {tag}")

        return tag


class ModelRegistry:
    """
    Simple model registry for versioning and tracking.
    """

    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.registry_path / "index.json"
        self.models = self._load_index()

    def _load_index(self) -> Dict:
        """Load model index."""
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Save model index."""
        with open(self.index_file, "w") as f:
            json.dump(self.models, f, indent=2)

    def register(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Register a new model version.

        Args:
            model_name: Name of the model
            version: Version string (e.g., "v1.0.0")
            model_path: Path to model file
            metadata: Additional metadata

        Returns:
            Model URI
        """
        import shutil

        # Create version directory
        version_dir = self.registry_path / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy model
        dest_path = version_dir / "model.pt"
        shutil.copy2(model_path, dest_path)

        # Save metadata
        model_info = {
            "name": model_name,
            "version": version,
            "path": str(dest_path),
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        with open(version_dir / "metadata.json", "w") as f:
            json.dump(model_info, f, indent=2)

        # Update index
        if model_name not in self.models:
            self.models[model_name] = {}

        self.models[model_name][version] = model_info
        self._save_index()

        print(f"✓ Registered {model_name}:{version}")
        return f"{model_name}:{version}"

    def get_model(self, model_name: str, version: Optional[str] = None) -> str:
        """
        Get path to model.

        Args:
            model_name: Name of the model
            version: Specific version, or None for latest

        Returns:
            Path to model file
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")

        if version is None:
            # Get latest version
            version = max(self.models[model_name].keys())

        if version not in self.models[model_name]:
            raise ValueError(f"Version {version} not found for {model_name}")

        model_info = self.models[model_name][version]
        return model_info["path"]

    def list_models(self) -> Dict[str, List[str]]:
        """List all registered models and versions."""
        return {name: list(versions.keys()) for name, versions in self.models.items()}


class ABTestManager:
    """
    Manage A/B testing for model deployment.
    """

    def __init__(self, test_config_path: str = "ab_tests.json"):
        self.config_path = Path(test_config_path)
        self.tests = self._load_tests()

    def _load_tests(self) -> Dict:
        """Load test configurations."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return json.load(f)
        return {}

    def _save_tests(self) -> None:
        """Save test configurations."""
        with open(self.config_path, "w") as f:
            json.dump(self.tests, f, indent=2)

    def create_test(
        self, test_name: str, model_a: str, model_b: str, traffic_split: float = 0.5
    ) -> str:
        """
        Create new A/B test.

        Args:
            test_name: Name of the test
            model_a: Control model (URI or path)
            model_b: Treatment model (URI or path)
            traffic_split: Fraction of traffic to model_b

        Returns:
            Test ID
        """
        test_id = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.tests[test_id] = {
            "name": test_name,
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,
            "created_at": datetime.now().isoformat(),
            "status": "active",
        }

        self._save_tests()
        print(f"✓ Created A/B test: {test_id}")
        return test_id

    def route_request(self, test_id: str, user_id: str) -> str:
        """
        Determine which model to use for a request.

        Args:
            test_id: Test ID
            user_id: User identifier for consistent routing

        Returns:
            Model path to use
        """
        import hashlib

        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.tests[test_id]

        # Hash user_id to get consistent assignment
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        assignment = (hash_val % 100) / 100

        if assignment < test["traffic_split"]:
            return test["model_b"]
        else:
            return test["model_a"]

    def end_test(self, test_id: str, winner: str) -> None:
        """
        End A/B test and declare winner.

        Args:
            test_id: Test ID
            winner: "a" or "b"
        """
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")

        self.tests[test_id]["status"] = "completed"
        self.tests[test_id]["winner"] = winner
        self.tests[test_id]["ended_at"] = datetime.now().isoformat()

        self._save_tests()
        print(f"✓ Ended test {test_id}, winner: {winner}")


class HealthMonitor:
    """
    Monitor model health and performance.
    """

    def __init__(self, metrics_path: str = "./monitoring/metrics.json"):
        self.metrics_path = Path(metrics_path)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = []

    def log_prediction(
        self, model_name: str, latency: float, input_shape: tuple, output_shape: tuple
    ) -> None:
        """Log a prediction for monitoring."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "latency_ms": latency,
            "input_shape": input_shape,
            "output_shape": output_shape,
        }

        self.metrics.append(metric)

        # Save periodically
        if len(self.metrics) % 100 == 0:
            self._save_metrics()

    def _save_metrics(self) -> None:
        """Save metrics to file."""
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        if not self.metrics:
            return {}

        latencies = [m["latency_ms"] for m in self.metrics]

        return {
            "total_requests": len(self.metrics),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        }

    def check_health(self, latency_threshold_ms: float = 100) -> Dict[str, Any]:
        """
        Check model health.

        Returns:
            Health status
        """
        stats = self.get_statistics()

        if not stats:
            return {"status": "unknown", "reason": "no metrics"}

        avg_latency = stats["avg_latency_ms"]

        if avg_latency > latency_threshold_ms:
            return {
                "status": "degraded",
                "reason": f"High latency: {avg_latency:.1f}ms",
                "statistics": stats,
            }

        return {"status": "healthy", "statistics": stats}


# Convenience functions
def deploy_docker(model_path: str, tag: str, port: int = 8000) -> str:
    """
    Quick deploy with Docker.

    Args:
        model_path: Path to model
        tag: Docker image tag
        port: Port to expose

    Returns:
        Container ID
    """
    import subprocess

    # Build
    builder = DockerBuilder()
    builder.build(tag, model_path)

    # Run
    cmd = ["docker", "run", "-d", "-p", f"{port}:{port}", tag]
    result = subprocess.run(cmd, capture_output=True, text=True)

    container_id = result.stdout.strip()
    print(f"✓ Deployed container: {container_id[:12]}")
    print(f"✓ API available at http://localhost:{port}")

    return container_id


def register_model(
    model_name: str,
    version: str,
    model_path: str,
    registry_path: str = "./model_registry",
) -> str:
    """Quick model registration."""
    registry = ModelRegistry(registry_path)
    return registry.register(model_name, version, model_path)


def check_model_health(metrics_path: str = "./monitoring/metrics.json") -> Dict:
    """Quick health check."""
    monitor = HealthMonitor(metrics_path)
    return monitor.check_health()
