"""
Experiment Tracking Module for fishstick

Integrates with popular experiment tracking tools:
- Weights & Biases (W&B)
- MLflow
- TensorBoard
- Neptune (optional)

Provides unified interface for logging metrics, parameters, artifacts, and models.
"""

from typing import Dict, Any, Optional, Union, List
import os
import json
from datetime import datetime
from pathlib import Path
import torch
from torch import nn


class BaseTracker:
    """Base class for experiment trackers."""

    def __init__(self, project_name: str, experiment_name: Optional[str] = None):
        self.project_name = project_name
        self.experiment_name = (
            experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.run_id = None
        self.config = {}

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        raise NotImplementedError

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics."""
        raise NotImplementedError

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact file."""
        raise NotImplementedError

    def log_model(self, model: nn.Module, model_name: str) -> None:
        """Log model."""
        raise NotImplementedError

    def finish(self) -> None:
        """Finish tracking."""
        raise NotImplementedError


class WandbTracker(BaseTracker):
    """
    Weights & Biases tracker integration.

    W&B is great for:
    - Real-time metric visualization
    - Hyperparameter sweeps
    - Model versioning
    - Collaboration
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(project_name, experiment_name)

        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb not installed. Run: pip install wandb")

        # Login if API key provided
        if api_key:
            self.wandb.login(key=api_key)

        # Initialize run
        self.run = self.wandb.init(
            project=project_name, name=experiment_name, reinit=True
        )
        self.run_id = self.run.id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to W&B."""
        self.wandb.config.update(params)
        self.config.update(params)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to W&B."""
        self.wandb.log(metrics, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact to W&B."""
        artifact = self.wandb.Artifact(
            name=artifact_path or Path(path).name,
            type="dataset" if "data" in path else "model",
        )
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def log_model(self, model: nn.Module, model_name: str) -> None:
        """Log PyTorch model to W&B."""
        # Save model temporarily
        temp_path = f"/tmp/{model_name}.pt"
        torch.save(model.state_dict(), temp_path)

        # Log as artifact
        self.log_artifact(temp_path, model_name)

        # Clean up
        os.remove(temp_path)

    def log_image(
        self, image_key: str, image: Any, caption: Optional[str] = None
    ) -> None:
        """Log image to W&B."""
        self.wandb.log({image_key: self.wandb.Image(image, caption=caption)})

    def log_table(
        self, table_name: str, data: List[List[Any]], columns: List[str]
    ) -> None:
        """Log table to W&B."""
        table = self.wandb.Table(data=data, columns=columns)
        self.wandb.log({table_name: table})

    def finish(self) -> None:
        """Finish W&B run."""
        self.run.finish()


class MLflowTracker(BaseTracker):
    """
    MLflow tracker integration.

    MLflow is great for:
    - Local experiment tracking
    - Model registry
    - Reproducibility
    - Open source
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ):
        super().__init__(project_name, experiment_name)

        try:
            import mlflow

            self.mlflow = mlflow
        except ImportError:
            raise ImportError("mlflow not installed. Run: pip install mlflow")

        # Set tracking URI if provided
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        experiment = self.mlflow.get_experiment_by_name(project_name)
        if experiment is None:
            experiment_id = self.mlflow.create_experiment(project_name)
        else:
            experiment_id = experiment.experiment_id

        # Start run
        self.run = self.mlflow.start_run(
            experiment_id=experiment_id, run_name=experiment_name
        )
        self.run_id = self.run.info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to MLflow."""
        for key, value in params.items():
            self.mlflow.log_param(key, value)
        self.config.update(params)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            self.mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact to MLflow."""
        self.mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_model(self, model: nn.Module, model_name: str) -> None:
        """Log PyTorch model to MLflow."""
        # Save model temporarily
        temp_path = f"/tmp/{model_name}"
        os.makedirs(temp_path, exist_ok=True)

        # Save model
        model_path = os.path.join(temp_path, "model.pt")
        torch.save(model.state_dict(), model_path)

        # Save architecture info
        arch_path = os.path.join(temp_path, "architecture.txt")
        with open(arch_path, "w") as f:
            f.write(str(model))

        # Log artifacts
        self.mlflow.log_artifacts(temp_path, artifact_path=model_name)

        # Clean up
        import shutil

        shutil.rmtree(temp_path)

    def finish(self) -> None:
        """Finish MLflow run."""
        self.mlflow.end_run()


class TensorBoardTracker(BaseTracker):
    """
    TensorBoard tracker integration.

    TensorBoard is great for:
    - Local visualization
    - No external dependencies
    - Built into PyTorch
    - Detailed model graphs
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        log_dir: str = "./runs",
    ):
        super().__init__(project_name, experiment_name)

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.SummaryWriter = SummaryWriter
        except ImportError:
            raise ImportError("tensorboard not installed. Run: pip install tensorboard")

        # Create log directory
        log_path = Path(log_dir) / project_name / self.experiment_name
        log_path.mkdir(parents=True, exist_ok=True)

        # Initialize writer
        self.writer = self.SummaryWriter(log_dir=str(log_path))
        self.run_id = self.experiment_name
        self.step = 0

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        # TensorBoard doesn't have native param logging, use hparams
        from torch.utils.tensorboard.summary import hparams

        # Convert to format TensorBoard expects
        hparams_dict = {}
        metrics_dict = {}

        for key, value in params.items():
            if isinstance(value, (int, float)):
                hparams_dict[key] = value
            else:
                hparams_dict[key] = str(value)

        self.writer.add_hparams(hparams_dict, metrics_dict)
        self.config.update(params)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to TensorBoard."""
        current_step = step if step is not None else self.step

        for key, value in metrics.items():
            self.writer.add_scalar(key, value, current_step)

        if step is None:
            self.step += 1

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """TensorBoard doesn't support artifacts directly."""
        print(f"Note: TensorBoard doesn't support artifacts. File saved at: {path}")

    def log_model(self, model: nn.Module, model_name: str) -> None:
        """Log model graph to TensorBoard."""
        # Add model graph (requires dummy input)
        print(f"Note: To log model graph, use writer.add_graph() with dummy input")

    def log_image(self, tag: str, image: Any, step: Optional[int] = None) -> None:
        """Log image to TensorBoard."""
        from torchvision.utils import make_grid

        if isinstance(image, torch.Tensor):
            self.writer.add_image(tag, image, step if step else self.step)

    def log_histogram(
        self, tag: str, values: torch.Tensor, step: Optional[int] = None
    ) -> None:
        """Log histogram of values."""
        self.writer.add_histogram(tag, values, step if step else self.step)

    def finish(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()

    def launch_dashboard(self, port: int = 6006) -> None:
        """Launch TensorBoard dashboard."""
        import subprocess

        subprocess.Popen(
            [
                "tensorboard",
                "--logdir",
                str(Path(self.writer.log_dir).parent.parent),
                "--port",
                str(port),
            ]
        )
        print(f"TensorBoard launched at http://localhost:{port}")


class MultiTracker:
    """
    Track experiments across multiple backends simultaneously.

    Example:
        tracker = MultiTracker(
            project_name="my_project",
            trackers=["wandb", "mlflow", "tensorboard"]
        )

        tracker.log_params({"lr": 0.001, "batch_size": 32})
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        trackers: List[str] = None,
        **kwargs,
    ):
        """
        Args:
            project_name: Name of the project
            experiment_name: Name of this experiment
            trackers: List of tracker names to use ["wandb", "mlflow", "tensorboard"]
            **kwargs: Additional arguments for specific trackers
        """
        self.trackers: Dict[str, BaseTracker] = {}

        if trackers is None:
            trackers = ["tensorboard"]  # Default to tensorboard (always works)

        for tracker_name in trackers:
            try:
                if tracker_name == "wandb":
                    self.trackers["wandb"] = WandbTracker(
                        project_name,
                        experiment_name,
                        api_key=kwargs.get("wandb_api_key"),
                    )
                elif tracker_name == "mlflow":
                    self.trackers["mlflow"] = MLflowTracker(
                        project_name,
                        experiment_name,
                        tracking_uri=kwargs.get("mlflow_uri"),
                    )
                elif tracker_name == "tensorboard":
                    self.trackers["tensorboard"] = TensorBoardTracker(
                        project_name,
                        experiment_name,
                        log_dir=kwargs.get("tensorboard_dir", "./runs"),
                    )
                else:
                    print(f"Unknown tracker: {tracker_name}")
            except Exception as e:
                print(f"Failed to initialize {tracker_name}: {e}")

        if not self.trackers:
            print("Warning: No trackers initialized. Logging to console only.")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log params to all trackers."""
        for name, tracker in self.trackers.items():
            try:
                tracker.log_params(params)
            except Exception as e:
                print(f"Failed to log params to {name}: {e}")

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to all trackers."""
        for name, tracker in self.trackers.items():
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                print(f"Failed to log metrics to {name}: {e}")

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact to all trackers."""
        for name, tracker in self.trackers.items():
            try:
                tracker.log_artifact(path, artifact_path)
            except Exception as e:
                print(f"Failed to log artifact to {name}: {e}")

    def log_model(self, model: nn.Module, model_name: str) -> None:
        """Log model to all trackers."""
        for name, tracker in self.trackers.items():
            try:
                tracker.log_model(model, model_name)
            except Exception as e:
                print(f"Failed to log model to {name}: {e}")

    def finish(self) -> None:
        """Finish all trackers."""
        for name, tracker in self.trackers.items():
            try:
                tracker.finish()
            except Exception as e:
                print(f"Failed to finish {name}: {e}")


class ExperimentLogger:
    """
    High-level experiment logger with automatic metric tracking.

    Automatically logs:
    - Training/validation metrics
    - Model checkpoints
    - System metrics (memory, GPU usage)
    - Training time
    """

    def __init__(
        self, tracker: Union[BaseTracker, MultiTracker], log_interval: int = 10
    ):
        self.tracker = tracker
        self.log_interval = log_interval
        self.start_time = datetime.now()
        self.step = 0

    def log_training_step(
        self, metrics: Dict[str, float], model: Optional[nn.Module] = None
    ) -> None:
        """Log a training step."""
        self.step += 1

        if self.step % self.log_interval == 0:
            self.tracker.log_metrics(metrics, step=self.step)

            # Log model periodically
            if model and self.step % (self.log_interval * 10) == 0:
                self.tracker.log_model(model, f"model_step_{self.step}")

    def log_epoch_end(
        self, epoch: int, metrics: Dict[str, float], model: Optional[nn.Module] = None
    ) -> None:
        """Log end of epoch."""
        # Prefix metrics with 'epoch/'
        epoch_metrics = {f"epoch/{k}": v for k, v in metrics.items()}
        epoch_metrics["epoch/number"] = epoch

        self.tracker.log_metrics(epoch_metrics, step=self.step)

        # Save model checkpoint
        if model:
            self.tracker.log_model(model, f"model_epoch_{epoch}")

    def log_system_metrics(self) -> None:
        """Log system metrics (CPU, memory)."""
        import psutil

        metrics = {
            "system/cpu_percent": psutil.cpu_percent(),
            "system/memory_percent": psutil.virtual_memory().percent,
        }

        self.tracker.log_metrics(metrics, step=self.step)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training run."""
        elapsed = datetime.now() - self.start_time

        return {
            "total_steps": self.step,
            "elapsed_time": str(elapsed),
            "steps_per_second": self.step / elapsed.total_seconds(),
        }

    def finish(self) -> None:
        """Finish logging."""
        # Log final summary
        summary = self.get_training_summary()
        self.tracker.log_params({"training_summary": summary})
        self.tracker.finish()


# Convenience functions
def create_tracker(
    project_name: str,
    experiment_name: Optional[str] = None,
    backend: str = "tensorboard",
    **kwargs,
) -> BaseTracker:
    """
    Create a tracker with specified backend.

    Args:
        project_name: Name of the project
        experiment_name: Name of the experiment
        backend: One of "wandb", "mlflow", "tensorboard"
        **kwargs: Additional arguments for the tracker

    Returns:
        Configured tracker instance
    """
    if backend == "wandb":
        return WandbTracker(project_name, experiment_name, **kwargs)
    elif backend == "mlflow":
        return MLflowTracker(project_name, experiment_name, **kwargs)
    elif backend == "tensorboard":
        return TensorBoardTracker(project_name, experiment_name, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def create_multi_tracker(
    project_name: str,
    experiment_name: Optional[str] = None,
    trackers: List[str] = None,
    **kwargs,
) -> MultiTracker:
    """
    Create a multi-backend tracker.

    Args:
        project_name: Name of the project
        experiment_name: Name of the experiment
        trackers: List of tracker names ["wandb", "mlflow", "tensorboard"]
        **kwargs: Additional arguments

    Returns:
        MultiTracker instance
    """
    return MultiTracker(project_name, experiment_name, trackers, **kwargs)
