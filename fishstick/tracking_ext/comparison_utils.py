"""
Comparison Utilities

Comprehensive utilities for comparing experiments and results.

Classes:
- ExperimentComparator: Compare experiment results
- MetricComparator: Compare metrics across experiments
- StatisticalTests: Statistical comparison utilities
- RankingCalculator: Calculate rankings
- ReportGenerator: Generate comparison reports
- TableFormatter: Format comparison tables
"""

from typing import Optional, Dict, List, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
from datetime import datetime


@dataclass
class ExperimentResult:
    """Container for experiment results."""

    experiment_name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get metric value.

        Args:
            name: Metric name
            default: Default value if not found

        Returns:
            Metric value
        """
        return self.metrics.get(name, default)


class ExperimentComparator:
    """Compare experiment results."""

    def __init__(self, experiments: Optional[List[ExperimentResult]] = None):
        """Initialize experiment comparator.

        Args:
            experiments: List of experiment results
        """
        self.experiments = experiments or []

    def add_experiment(self, experiment: ExperimentResult) -> None:
        """Add an experiment result.

        Args:
            experiment: Experiment result to add
        """
        self.experiments.append(experiment)

    def compare_metrics(
        self, metric_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare metrics across experiments.

        Args:
            metric_names: List of metric names to compare

        Returns:
            Dictionary of metric comparisons
        """
        if not metric_names:
            metric_names = self._get_all_metric_names()

        comparisons = {}
        for metric in metric_names:
            values = {
                exp.experiment_name: exp.get_metric(metric) for exp in self.experiments
            }
            comparisons[metric] = values

        return comparisons

    def _get_all_metric_names(self) -> List[str]:
        """Get all unique metric names.

        Returns:
            List of metric names
        """
        metric_names = set()
        for exp in self.experiments:
            metric_names.update(exp.metrics.keys())
        return sorted(list(metric_names))

    def get_best_experiment(
        self,
        metric_name: str,
        mode: str = "max",
    ) -> Optional[ExperimentResult]:
        """Get best experiment by metric.

        Args:
            metric_name: Metric to compare
            mode: 'max' or 'min'

        Returns:
            Best experiment or None
        """
        if not self.experiments:
            return None

        best_exp = None
        best_value = float("-inf") if mode == "max" else float("inf")

        for exp in self.experiments:
            value = exp.get_metric(metric_name)
            if mode == "max":
                if value > best_value:
                    best_value = value
                    best_exp = exp
            else:
                if value < best_value:
                    best_value = value
                    best_exp = exp

        return best_exp

    def get_rankings(
        self, metric_name: str, mode: str = "max"
    ) -> List[Tuple[int, str, float]]:
        """Get experiment rankings by metric.

        Args:
            metric_name: Metric to rank by
            mode: 'max' or 'min'

        Returns:
            List of (rank, experiment_name, value)
        """
        values = [
            (exp.experiment_name, exp.get_metric(metric_name))
            for exp in self.experiments
        ]
        values.sort(key=lambda x: x[1], reverse=(mode == "max"))

        return [(i + 1, name, value) for i, (name, value) in enumerate(values)]

    def get_improvement(
        self,
        baseline_exp: str,
        comparison_exp: str,
        metric_name: str,
    ) -> Dict[str, float]:
        """Calculate improvement of comparison over baseline.

        Args:
            baseline_exp: Baseline experiment name
            comparison_exp: Comparison experiment name
            metric_name: Metric to compare

        Returns:
            Dictionary with absolute and relative improvement
        """
        baseline = next(
            (e for e in self.experiments if e.experiment_name == baseline_exp), None
        )
        comparison = next(
            (e for e in self.experiments if e.experiment_name == comparison_exp), None
        )

        if not baseline or not comparison:
            return {"absolute": 0.0, "relative": 0.0, "percentage": 0.0}

        baseline_value = baseline.get_metric(metric_name)
        comparison_value = comparison.get_metric(metric_name)

        absolute = comparison_value - baseline_value
        relative = absolute / (baseline_value if baseline_value != 0 else 1e-10)
        percentage = relative * 100

        return {
            "absolute": absolute,
            "relative": relative,
            "percentage": percentage,
        }


class MetricComparator:
    """Compare metrics with statistical tests."""

    @staticmethod
    def compare_runs(
        runs: List[List[float]],
        names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare multiple runs statistically.

        Args:
            runs: List of run value lists
            names: Optional names for runs

        Returns:
            Statistical comparison results
        """
        names = names or [f"Run {i + 1}" for i in range(len(runs))]

        results = {
            "names": names,
            "means": [np.mean(run) for run in runs],
            "stds": [np.std(run) for run in runs],
            "medians": [np.median(run) for run in runs],
            "min": [np.min(run) for run in runs],
            "max": [np.max(run) for run in runs],
        }

        return results

    @staticmethod
    def pairwise_tests(
        runs: List[List[float]],
        test: str = "t-test",
    ) -> List[Dict[str, Any]]:
        """Perform pairwise statistical tests.

        Args:
            runs: List of run value lists
            test: Test type ('t-test', 'mann-whitney', 'welch')

        Returns:
            List of test results
        """
        from scipy import stats

        results = []
        n_runs = len(runs)

        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                if test == "t-test":
                    statistic, pvalue = stats.ttest_ind(runs[i], runs[j])
                elif test == "mann-whitney":
                    statistic, pvalue = stats.mannwhitneyu(runs[i], runs[j])
                elif test == "welch":
                    statistic, pvalue = stats.ttest_ind(
                        runs[i], runs[j], equal_var=False
                    )
                else:
                    continue

                results.append(
                    {
                        "run_i": i,
                        "run_j": j,
                        "test": test,
                        "statistic": float(statistic),
                        "pvalue": float(pvalue),
                        "significant": float(pvalue) < 0.05,
                    }
                )

        return results


class RankingCalculator:
    """Calculate various rankings."""

    @staticmethod
    def calculate_weighted_ranking(
        metrics: Dict[str, Dict[str, float]],
        weights: Optional[Dict[str, float]] = None,
        modes: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[int, str, float]]:
        """Calculate weighted ranking across metrics.

        Args:
            metrics: Dict of metric_name to {exp_name: value}
            weights: Weights for each metric
            modes: Mode for each metric ('max' or 'min')

        Returns:
            List of (rank, experiment_name, score)
        """
        if not metrics:
            return []

        weights = weights or {k: 1.0 for k in metrics.keys()}
        modes = modes or {k: "max" for k in metrics.keys()}

        exp_names = set()
        for metric_values in metrics.values():
            exp_names.update(metric_values.keys())

        scores = {}
        for exp_name in exp_names:
            score = 0.0
            for metric_name, metric_values in metrics.items():
                value = metric_values.get(exp_name, 0.0)
                weight = weights.get(metric_name, 1.0)
                mode = modes.get(metric_name, "max")

                if mode == "max":
                    normalized = value
                else:
                    normalized = -value

                score += weight * normalized

            scores[exp_name] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(i + 1, name, score) for i, (name, score) in enumerate(ranked)]

    @staticmethod
    def calculate_percentile_ranks(
        metric_values: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate percentile ranks.

        Args:
            metric_values: Dictionary of name to value

        Returns:
            Dictionary of name to percentile rank
        """
        values = list(metric_values.values())
        sorted_values = sorted(values)

        ranks = {}
        for name, value in metric_values.items():
            percentile = (
                sum(1 for v in sorted_values if v < value) / len(sorted_values) * 100
            )
            ranks[name] = percentile

        return ranks


class StatisticalTests:
    """Statistical tests for comparison."""

    @staticmethod
    def t_test(
        values1: List[float],
        values2: List[float],
        equal_var: bool = True,
    ) -> Dict[str, float]:
        """Perform t-test.

        Args:
            values1: First set of values
            values2: Second set of values
            equal_var: Assume equal variance

        Returns:
            Test results
        """
        from scipy import stats

        statistic, pvalue = stats.ttest_ind(values1, values2, equal_var=equal_var)

        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": float(pvalue) < 0.05,
        }

    @staticmethod
    def mann_whitney_u(
        values1: List[float],
        values2: List[float],
    ) -> Dict[str, float]:
        """Perform Mann-Whitney U test.

        Args:
            values1: First set of values
            values2: Second set of values

        Returns:
            Test results
        """
        from scipy import stats

        statistic, pvalue = stats.mannwhitneyu(values1, values2)

        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": float(pvalue) < 0.05,
        }

    @staticmethod
    def bootstrap_confidence_interval(
        values: List[float],
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval.

        Args:
            values: Values to compute CI for
            n_bootstrap: Number of bootstrap samples
            ci: Confidence level

        Returns:
            (lower_bound, upper_bound)
        """
        np.random.seed(42)
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = (1 - ci) / 2
        lower = np.percentile(bootstrap_means, alpha * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

        return float(lower), float(upper)

    @staticmethod
    def effect_size(
        values1: List[float],
        values2: List[float],
    ) -> Dict[str, float]:
        """Calculate effect size (Cohen's d).

        Args:
            values1: First set of values
            values2: Second set of values

        Returns:
            Effect size metrics
        """
        mean1, mean2 = np.mean(values1), np.mean(values2)
        std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
        n1, n2 = len(values1), len(values2)

        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / (pooled_std if pooled_std > 0 else 1e-10)

        return {
            "cohens_d": float(cohens_d),
            "interpretation": "small"
            if abs(cohens_d) < 0.5
            else "medium"
            if abs(cohens_d) < 0.8
            else "large",
        }


class TableFormatter:
    """Format comparison tables."""

    @staticmethod
    def format_table(
        data: Dict[str, Dict[str, float]],
        headers: Optional[List[str]] = None,
        precision: int = 4,
    ) -> str:
        """Format data as table.

        Args:
            data: Dictionary of row_name to {col_name: value}
            headers: Column headers
            precision: Decimal precision

        Returns:
            Formatted table string
        """
        if not data:
            return ""

        if headers is None:
            headers = list(next(iter(data.values())).keys())

        col_widths = {h: len(h) for h in headers}
        for row_name, row_data in data.items():
            col_widths[row_name] = len(row_name)
            for col in headers:
                value_str = f"{row_data.get(col, 0):.{precision}f}"
                col_widths[col] = max(col_widths[col], len(value_str))

        lines = []

        header_line = (
            " " * 10 + " | " + " | ".join(h.ljust(col_widths[h]) for h in headers)
        )
        lines.append(header_line)
        lines.append("-" * len(header_line))

        for row_name, row_data in data.items():
            row_line = row_name.ljust(10) + " | "
            row_line += " | ".join(
                f"{row_data.get(col, 0):.{precision}f}".ljust(col_widths[col])
                for col in headers
            )
            lines.append(row_line)

        return "\n".join(lines)

    @staticmethod
    def format_markdown(
        data: Dict[str, Dict[str, float]],
        headers: Optional[List[str]] = None,
        precision: int = 4,
    ) -> str:
        """Format data as markdown table.

        Args:
            data: Dictionary of row_name to {col_name: value}
            headers: Column headers
            precision: Decimal precision

        Returns:
            Markdown table string
        """
        if not data:
            return ""

        if headers is None:
            headers = list(next(iter(data.values())).keys())

        lines = []

        lines.append("| Experiment | " + " | ".join(headers) + " |")
        lines.append(
            "|------------|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|"
        )

        for row_name, row_data in data.items():
            values = [f"{row_data.get(col, 0):.{precision}f}" for col in headers]
            lines.append(f"| {row_name} | " + " | ".join(values) + " |")

        return "\n".join(lines)


class ReportGenerator:
    """Generate comparison reports."""

    def __init__(self, comparator: ExperimentComparator):
        """Initialize report generator.

        Args:
            comparator: Experiment comparator
        """
        self.comparator = comparator

    def generate_text_report(
        self,
        metrics: Optional[List[str]] = None,
    ) -> str:
        """Generate text report.

        Args:
            metrics: Metrics to include

        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("EXPERIMENT COMPARISON REPORT")
        lines.append("=" * 60)
        lines.append("")

        comparisons = self.comparator.compare_metrics(metrics)

        for metric, values in comparisons.items():
            lines.append(f"Metric: {metric}")
            lines.append("-" * 40)

            best_exp = self.comparator.get_best_experiment(metric)
            if best_exp:
                lines.append(
                    f"Best: {best_exp.experiment_name} = {best_exp.get_metric(metric):.4f}"
                )

            rankings = self.comparator.get_rankings(metric)
            for rank, name, value in rankings:
                lines.append(f"  {rank}. {name}: {value:.4f}")

            lines.append("")

        return "\n".join(lines)

    def generate_markdown_report(
        self,
        metrics: Optional[List[str]] = None,
    ) -> str:
        """Generate markdown report.

        Args:
            metrics: Metrics to include

        Returns:
            Markdown report string
        """
        lines = []
        lines.append("# Experiment Comparison Report")
        lines.append("")

        comparisons = self.comparator.compare_metrics(metrics)

        for metric, values in comparisons.items():
            lines.append(f"## {metric}")
            lines.append("")

            rankings = self.comparator.get_rankings(metric)
            lines.append("| Rank | Experiment | Value |")
            lines.append("|------|------------|-------|")

            for rank, name, value in rankings:
                lines.append(f"| {rank} | {name} | {value:.4f} |")

            lines.append("")

        return "\n".join(lines)

    def save_report(
        self,
        path: Union[str, Path],
        format: str = "markdown",
    ) -> None:
        """Save report to file.

        Args:
            path: Output path
            format: 'markdown' or 'text'
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            report = self.generate_markdown_report()
        else:
            report = self.generate_text_report()

        with open(path, "w") as f:
            f.write(report)


class ExperimentDatabase:
    """Database for storing and querying experiments."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize experiment database.

        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = storage_path or Path("experiments_db")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, ExperimentResult] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all experiments from storage."""
        for exp_file in self.storage_path.glob("*.json"):
            with open(exp_file, "r") as f:
                data = json.load(f)
                exp = ExperimentResult(
                    experiment_name=data["experiment_name"],
                    config=data.get("config", {}),
                    metrics=data.get("metrics", {}),
                    metadata=data.get("metadata", {}),
                    timestamp=data.get("timestamp"),
                )
                self.experiments[exp.experiment_name] = exp

    def add(self, experiment: ExperimentResult) -> None:
        """Add experiment to database.

        Args:
            experiment: Experiment to add
        """
        self.experiments[experiment.experiment_name] = experiment
        self._save_experiment(experiment)

    def _save_experiment(self, experiment: ExperimentResult) -> None:
        """Save experiment to storage.

        Args:
            experiment: Experiment to save
        """
        data = {
            "experiment_name": experiment.experiment_name,
            "config": experiment.config,
            "metrics": experiment.metrics,
            "metadata": experiment.metadata,
            "timestamp": experiment.timestamp or datetime.now().timestamp(),
        }

        filename = f"{experiment.experiment_name}.json"
        with open(self.storage_path / filename, "w") as f:
            json.dump(data, f, indent=2)

    def get(self, name: str) -> Optional[ExperimentResult]:
        """Get experiment by name.

        Args:
            name: Experiment name

        Returns:
            Experiment or None
        """
        return self.experiments.get(name)

    def query(
        self,
        filter_fn: Callable[[ExperimentResult], bool],
    ) -> List[ExperimentResult]:
        """Query experiments by filter function.

        Args:
            filter_fn: Filter function

        Returns:
            List of matching experiments
        """
        return [exp for exp in self.experiments.values() if filter_fn(exp)]

    def get_best_by_metric(
        self,
        metric_name: str,
        mode: str = "max",
        n: int = 1,
    ) -> List[ExperimentResult]:
        """Get best experiments by metric.

        Args:
            metric_name: Metric name
            mode: 'max' or 'min'
            n: Number of experiments

        Returns:
            List of best experiments
        """
        values = [
            (
                exp,
                exp.get_metric(
                    metric_name, float("-inf") if mode == "max" else float("inf")
                ),
            )
            for exp in self.experiments.values()
        ]
        values.sort(key=lambda x: x[1], reverse=(mode == "max"))

        return [exp for exp, _ in values[:n]]

    def list_experiments(self) -> List[str]:
        """List all experiment names.

        Returns:
            List of experiment names
        """
        return sorted(self.experiments.keys())
