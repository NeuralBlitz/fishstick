"""
Experiment Tracking System

Track experiments, log metrics, compare runs, and manage artifacts.
"""

from typing import Dict, List, Any, Optional, Union
import torch
import torch.nn as nn
import json
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import pickle
import warnings


class Experiment:
    """
    Single experiment tracker.

    Tracks metrics, parameters, artifacts, and metadata for one training run.

    Example:
        >>> exp = Experiment(name="resnet50_baseline", log_dir="experiments")
        >>> exp.log_params({"lr": 0.001, "batch_size": 32})
        >>>
        >>> for epoch in range(10):
        ...     train_loss = train_epoch(...)
        ...     val_acc = validate(...)
        ...     exp.log_metrics({"train_loss": train_loss, "val_acc": val_acc}, step=epoch)
        >>>
        >>> exp.save_model(model, "final_model.pt")
        >>> exp.finish()
    """

    def __init__(
        self,
        name: Optional[str] = None,
        log_dir: str = "experiments",
        tags: Optional[List[str]] = None,
        config: Optional[Dict] = None,
    ):
        self.name = name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(log_dir) / self.name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.tags = tags or []
        self.config = config or {}

        self.metrics = defaultdict(list)
        self.params = {}
        self.artifacts = []

        self.start_time = time.time()
        self.end_time = None

        self.run_id = self._generate_run_id()

        # Save initial metadata
        self._save_metadata()

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = str(time.time())
        name_hash = hashlib.md5(self.name.encode()).hexdigest()[:8]
        return f"{name_hash}_{timestamp[-6:]}"

    def _save_metadata(self):
        """Save experiment metadata."""
        metadata = {
            "run_id": self.run_id,
            "name": self.name,
            "tags": self.tags,
            "config": self.config,
            "start_time": self.start_time,
            "log_dir": str(self.log_dir),
        }

        with open(self.log_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self.params.update(params)

        # Save params
        with open(self.log_dir / "params.json", "w") as f:
            json.dump(self.params, f, indent=2, default=str)

    def log_metrics(
        self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None
    ):
        """Log metrics at a given step."""
        timestamp = time.time() - self.start_time

        for key, value in metrics.items():
            self.metrics[key].append(
                {"value": value, "step": step, "timestamp": timestamp}
            )

        # Save metrics
        self._save_metrics()

    def _save_metrics(self):
        """Save metrics to file."""
        metrics_dict = {}
        for key, values in self.metrics.items():
            metrics_dict[key] = {
                "values": [v["value"] for v in values],
                "steps": [v["step"] for v in values],
                "timestamps": [v["timestamp"] for v in values],
            }

        with open(self.log_dir / "metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)

    def log_artifact(self, path: Union[str, Path], artifact_type: str = "generic"):
        """Log an artifact file."""
        path = Path(path)

        if path.exists():
            # Copy to experiment directory
            dest = self.log_dir / "artifacts" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)

            if path.is_file():
                import shutil

                shutil.copy2(path, dest)

            self.artifacts.append(
                {
                    "path": str(dest),
                    "original_path": str(path),
                    "type": artifact_type,
                    "timestamp": time.time(),
                }
            )

            # Save artifacts list
            with open(self.log_dir / "artifacts.json", "w") as f:
                json.dump(self.artifacts, f, indent=2)

    def save_model(
        self,
        model: nn.Module,
        filename: str = "model.pt",
        metadata: Optional[Dict] = None,
    ):
        """Save model checkpoint."""
        filepath = self.log_dir / "checkpoints" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "params": self.params,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }

        torch.save(checkpoint, filepath)
        self.log_artifact(filepath, "model")

    def log_text(self, text: str, filename: str = "notes.txt"):
        """Log text notes."""
        filepath = self.log_dir / filename
        with open(filepath, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
            f.write(text)
            f.write("\n\n")

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}

        for key, values in self.metrics.items():
            vals = [v["value"] for v in values if isinstance(v["value"], (int, float))]
            if vals:
                summary[key] = {
                    "min": min(vals),
                    "max": max(vals),
                    "mean": sum(vals) / len(vals),
                    "final": vals[-1],
                }

        return summary

    def finish(self, status: str = "completed"):
        """Finish the experiment."""
        self.end_time = time.time()

        # Save final summary
        summary = {
            "status": status,
            "duration": self.end_time - self.start_time,
            "metrics_summary": self.get_metrics_summary(),
            "end_time": self.end_time,
        }

        with open(self.log_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


class ExperimentTracker:
    """
    Track multiple experiments and compare results.

    Args:
        log_dir: Root directory for all experiments

    Example:
        >>> tracker = ExperimentTracker(log_dir="my_experiments")
        >>>
        >>> # Create new experiment
        >>> exp = tracker.create_experiment(name="run_1", tags=["resnet", "baseline"])
        >>>
        >>> # Get all experiments
        >>> all_exps = tracker.list_experiments()
        >>>
        >>> # Compare experiments
        >>> comparison = tracker.compare_experiments([exp1.run_id, exp2.run_id])
    """

    def __init__(self, log_dir: str = "experiments"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.experiments: Dict[str, Experiment] = {}
        self._load_existing_experiments()

    def _load_existing_experiments(self):
        """Load existing experiments from log directory."""
        for exp_dir in self.log_dir.iterdir():
            if exp_dir.is_dir() and (exp_dir / "metadata.json").exists():
                try:
                    with open(exp_dir / "metadata.json", "r") as f:
                        metadata = json.load(f)

                    # Don't fully load, just track existence
                    self.experiments[metadata["run_id"]] = None
                except:
                    pass

    def create_experiment(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict] = None,
    ) -> Experiment:
        """Create a new experiment."""
        exp = Experiment(name=name, log_dir=str(self.log_dir), tags=tags, config=config)
        self.experiments[exp.run_id] = exp
        return exp

    def get_experiment(self, run_id: str) -> Optional[Experiment]:
        """Get experiment by run ID."""
        if run_id in self.experiments:
            if self.experiments[run_id] is None:
                # Lazy load
                for exp_dir in self.log_dir.iterdir():
                    if exp_dir.is_dir():
                        try:
                            with open(exp_dir / "metadata.json", "r") as f:
                                metadata = json.load(f)
                            if metadata["run_id"] == run_id:
                                exp = Experiment.__new__(Experiment)
                                exp.log_dir = exp_dir
                                exp.run_id = run_id
                                exp.name = metadata["name"]
                                exp.tags = metadata["tags"]
                                exp.config = metadata["config"]

                                # Load metrics
                                with open(exp_dir / "metrics.json", "r") as f:
                                    metrics_data = json.load(f)
                                    exp.metrics = defaultdict(list)
                                    for key, data in metrics_data.items():
                                        for i, value in enumerate(data["values"]):
                                            exp.metrics[key].append(
                                                {
                                                    "value": value,
                                                    "step": data["steps"][i],
                                                    "timestamp": data["timestamps"][i],
                                                }
                                            )

                                self.experiments[run_id] = exp
                                return exp
                        except:
                            pass
            else:
                return self.experiments[run_id]
        return None

    def list_experiments(
        self, tags: Optional[List[str]] = None, status: Optional[str] = None
    ) -> List[Dict]:
        """List all experiments with optional filtering."""
        experiments = []

        for run_id in self.experiments.keys():
            exp = self.get_experiment(run_id)
            if exp is None:
                continue

            # Filter by tags
            if tags and not any(tag in exp.tags for tag in tags):
                continue

            # Check status
            summary_path = exp.log_dir / "summary.json"
            exp_status = "running"
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                    exp_status = summary.get("status", "unknown")

            if status and exp_status != status:
                continue

            experiments.append(
                {
                    "run_id": run_id,
                    "name": exp.name,
                    "tags": exp.tags,
                    "status": exp_status,
                    "created": datetime.fromtimestamp(exp.start_time).isoformat(),
                }
            )

        return experiments

    def compare_experiments(
        self, run_ids: List[str], metrics: Optional[List[str]] = None
    ) -> Dict:
        """Compare multiple experiments."""
        comparison = {}

        for run_id in run_ids:
            exp = self.get_experiment(run_id)
            if exp is None:
                continue

            exp_data = {
                "name": exp.name,
                "params": exp.params,
                "metrics_summary": exp.get_metrics_summary(),
            }

            # Filter metrics if specified
            if metrics:
                exp_data["metrics_summary"] = {
                    k: v for k, v in exp_data["metrics_summary"].items() if k in metrics
                }

            comparison[run_id] = exp_data

        return comparison

    def get_best_experiment(
        self,
        metric: str = "val_accuracy",
        mode: str = "max",
        tags: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """Get best experiment based on a metric."""
        experiments = self.list_experiments(tags=tags, status="completed")

        if not experiments:
            return None

        best_exp = None
        best_score = float("-inf") if mode == "max" else float("inf")

        for exp_info in experiments:
            exp = self.get_experiment(exp_info["run_id"])
            if exp is None:
                continue

            summary = exp.get_metrics_summary()
            if metric in summary:
                score = summary[metric].get(
                    "max" if mode == "max" else "min", summary[metric]["final"]
                )

                if mode == "max" and score > best_score:
                    best_score = score
                    best_exp = exp_info
                elif mode == "min" and score < best_score:
                    best_score = score
                    best_exp = exp_info

        if best_exp:
            best_exp["best_score"] = best_score
            best_exp["metric"] = metric

        return best_exp

    def delete_experiment(self, run_id: str):
        """Delete an experiment and its files."""
        exp = self.get_experiment(run_id)
        if exp:
            import shutil

            shutil.rmtree(exp.log_dir)
            del self.experiments[run_id]

    def generate_report(self, output_file: str = "report.html"):
        """Generate HTML report of all experiments."""
        experiments = self.list_experiments()

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .tag { background-color: #e7f3ff; padding: 2px 6px; border-radius: 3px; margin: 2px; }
            </style>
        </head>
        <body>
            <h1>Experiment Report</h1>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Run ID</th>
                    <th>Tags</th>
                    <th>Status</th>
                    <th>Created</th>
                </tr>
        """

        for exp in experiments:
            tags_html = "".join([f'<span class="tag">{t}</span>' for t in exp["tags"]])
            html += f"""
                <tr>
                    <td>{exp["name"]}</td>
                    <td>{exp["run_id"][:8]}...</td>
                    <td>{tags_html}</td>
                    <td>{exp["status"]}</td>
                    <td>{exp["created"]}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        output_path = self.log_dir / output_file
        with open(output_path, "w") as f:
            f.write(html)

        print(f"Report saved to {output_path}")


__all__ = [
    "Experiment",
    "ExperimentTracker",
]
