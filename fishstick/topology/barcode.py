"""
Persistence Barcode Analysis and Statistics.

Provides tools for analyzing, comparing, and computing statistics
on persistence barcodes for topological data analysis.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import torch
from torch import Tensor
import numpy as np
from scipy import stats as scipy_stats
from scipy.signal import find_peaks


@dataclass
class BarcodeInterval:
    """Represents an interval in a persistence barcode."""

    birth: float
    death: float
    dimension: int
    multiplicity: int = 1

    @property
    def persistence(self) -> float:
        """Lifetime of the feature."""
        return self.death - self.birth if self.death != float("inf") else float("inf")

    @property
    def lifetime(self) -> float:
        """Alias for persistence."""
        return self.persistence

    @property
    def midpoint(self) -> float:
        """Midpoint of the interval."""
        if self.death == float("inf"):
            return self.birth
        return (self.birth + self.death) / 2


@dataclass
class PersistenceBarcode:
    """
    Persistence Barcode Representation.

    A barcode is a collection of intervals representing
    the lifespan of topological features across filtration.
    """

    dimension: int
    intervals: List[BarcodeInterval] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.intervals)

    def __iter__(self):
        return iter(self.intervals)

    @property
    def births(self) -> Tensor:
        return torch.tensor([i.birth for i in self.intervals])

    @property
    def deaths(self) -> Tensor:
        return torch.tensor(
            [
                i.death if i.death != float("inf") else float("max")
                for i in self.intervals
            ]
        )

    @property
    def persistences(self) -> Tensor:
        return torch.tensor(
            [
                i.persistence if i.persistence != float("inf") else float("max")
                for i in self.intervals
            ]
        )

    def filter_by_persistence(
        self,
        min_persistence: float,
    ) -> "PersistenceBarcode":
        """Filter intervals by minimum persistence."""
        filtered = [
            i
            for i in self.intervals
            if i.persistence != float("inf") and i.persistence >= min_persistence
        ]
        return PersistenceBarcode(self.dimension, filtered)

    def get_longest_intervals(
        self,
        n: int = 10,
    ) -> List[BarcodeInterval]:
        """Get the n longest intervals."""
        finite_intervals = [i for i in self.intervals if i.persistence != float("inf")]
        sorted_intervals = sorted(
            finite_intervals,
            key=lambda x: x.persistence,
            reverse=True,
        )
        return sorted_intervals[:n]

    def to_tensor(self) -> Tensor:
        """Convert to tensor representation."""
        data = []
        for interval in self.intervals:
            death_val = interval.death if interval.death != float("inf") else -1.0
            data.append([interval.birth, death_val, interval.dimension])
        return torch.tensor(data, dtype=torch.float32)


class BarcodeAnalyzer:
    """
    Analyzer for Persistence Barcodes.

    Provides statistical analysis and metrics for
    persistence barcodes.
    """

    def __init__(self):
        pass

    def compute_statistics(
        self,
        barcode: PersistenceBarcode,
    ) -> Dict[str, float]:
        """
        Compute statistics for a barcode.

        Args:
            barcode: Input persistence barcode

        Returns:
            Dictionary of statistics
        """
        if len(barcode) == 0:
            return {
                "n_intervals": 0,
                "mean_persistence": 0.0,
                "std_persistence": 0.0,
                "max_persistence": 0.0,
                "total_persistence": 0.0,
            }

        persistences = []
        for interval in barcode.intervals:
            if interval.persistence != float("inf"):
                persistences.append(interval.persistence)

        if len(persistences) == 0:
            return {
                "n_intervals": len(barcode),
                "mean_persistence": 0.0,
                "std_persistence": 0.0,
                "max_persistence": 0.0,
                "total_persistence": 0.0,
            }

        persistences_np = np.array(persistences)

        return {
            "n_intervals": len(barcode),
            "mean_persistence": float(np.mean(persistences_np)),
            "std_persistence": float(np.std(persistences_np)),
            "max_persistence": float(np.max(persistences_np)),
            "min_persistence": float(np.min(persistences_np)),
            "median_persistence": float(np.median(persistences_np)),
            "total_persistence": float(np.sum(persistences_np)),
            "q25_persistence": float(np.percentile(persistences_np, 25)),
            "q75_persistence": float(np.percentile(persistences_np, 75)),
        }

    def compute_lifespan_distribution(
        self,
        barcode: PersistenceBarcode,
        n_bins: int = 50,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute histogram of persistence values.

        Args:
            barcode: Input barcode
            n_bins: Number of histogram bins

        Returns:
            Tuple of (bin_centers, frequencies)
        """
        persistences = []
        for interval in barcode.intervals:
            if interval.persistence != float("inf"):
                persistences.append(interval.persistence)

        if len(persistences) == 0:
            return torch.zeros(n_bins), torch.zeros(n_bins)

        hist, edges = np.histogram(persistences, bins=n_bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2

        return torch.tensor(bin_centers), torch.tensor(hist)

    def find_significant_intervals(
        self,
        barcode: PersistenceBarcode,
        method: str = "elbow",
        threshold: Optional[float] = None,
    ) -> List[BarcodeInterval]:
        """
        Find statistically significant intervals.

        Args:
            barcode: Input barcode
            method: Method for finding significant intervals
            threshold: Manual threshold (if applicable)

        Returns:
            List of significant intervals
        """
        if method == "threshold" and threshold is not None:
            return [
                i
                for i in barcode.intervals
                if i.persistence >= threshold and i.persistence != float("inf")
            ]

        if method == "elbow":
            finite_intervals = [
                i for i in barcode.intervals if i.persistence != float("inf")
            ]

            if len(finite_intervals) < 3:
                return finite_intervals

            persistences = sorted([i.persistence for i in finite_intervals])

            if len(persistences) < 3:
                return finite_intervals

            diffs = np.diff(persistences)
            elbow_idx = np.argmax(diffs) + 1

            threshold = persistences[elbow_idx]

            return [i for i in finite_intervals if i.persistence >= threshold]

        if method == "outlier":
            persistences = []
            for interval in barcode.intervals:
                if interval.persistence != float("inf"):
                    persistences.append(interval.persistence)

            if len(persistences) < 4:
                return []

            q1 = np.percentile(persistences, 25)
            q3 = np.percentile(persistences, 75)
            iqr = q3 - q1
            upper_threshold = q3 + 1.5 * iqr

            return [
                i
                for i in barcode.intervals
                if i.persistence > upper_threshold and i.persistence != float("inf")
            ]

        return []

    def compute_betti_curve(
        self,
        barcode: PersistenceBarcode,
        resolution: int = 100,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute Betti curve from barcode.

        Args:
            barcode: Input barcode
            resolution: Number of sample points

        Returns:
            Tuple of (filtration_values, betti_numbers)
        """
        if len(barcode) == 0:
            return torch.zeros(resolution), torch.zeros(resolution)

        max_filtration = 0.0
        for interval in barcode.intervals:
            if interval.death != float("inf"):
                max_filtration = max(max_filtration, interval.death)

        filtration_values = torch.linspace(0, max_filtration, resolution)
        betti_numbers = []

        for t in filtration_values:
            count = sum(
                1
                for interval in barcode.intervals
                if interval.birth <= t < interval.death
            )
            betti_numbers.append(count)

        return filtration_values, torch.tensor(betti_numbers)

    def compute_persistence_entropy(
        self,
        barcode: PersistenceBarcode,
    ) -> float:
        """
        Compute persistence entropy of barcode.

        Args:
            barcode: Input barcode

        Returns:
            Entropy value
        """
        persistences = []
        total = 0.0

        for interval in barcode.intervals:
            if interval.persistence != float("inf") and interval.persistence > 0:
                persistences.append(interval.persistence)
                total += interval.persistence

        if len(persistences) == 0 or total == 0:
            return 0.0

        probabilities = [p / total for p in persistences]

        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probabilities)

        return entropy


class BarcodeComparator:
    """
    Compare Persistence Barcodes.

    Provides methods for comparing and computing
    distances between barcodes.
    """

    def __init__(self):
        pass

    def wasserstein_distance(
        self,
        barcode1: PersistenceBarcode,
        barcode2: PersistenceBarcode,
        p: float = 2.0,
    ) -> float:
        """
        Compute Wasserstein distance between barcodes.

        Args:
            barcode1: First barcode
            barcode2: Second barcode
            p: Exponent for Wasserstein distance

        Returns:
            Distance value
        """
        intervals1 = [
            (i.birth, i.death)
            for i in barcode1.intervals
            if i.persistence != float("inf")
        ]
        intervals2 = [
            (i.birth, i.death)
            for i in barcode2.intervals
            if i.persistence != float("inf")
        ]

        if len(intervals1) == 0:
            intervals1 = [(0, 0)]
        if len(intervals2) == 0:
            intervals2 = [(0, 0)]

        return self._compute_hausdorff_distance(intervals1, intervals2, p)

    def _compute_hausdorff_distance(
        self,
        intervals1: List[Tuple[float, float]],
        intervals2: List[Tuple[float, float]],
        p: float,
    ) -> float:
        """Compute generalized Hausdorff distance."""
        n1 = len(intervals1)
        n2 = len(intervals2)
        max_n = max(n1, n2)

        if max_n == 0:
            return 0.0

        dists = []

        for b1, d1 in intervals1:
            min_dist = (
                min(
                    (abs(b1 - b2) ** p + abs(d1 - d2) ** p) ** (1 / p)
                    for b2, d2 in intervals2
                )
                if intervals2
                else float("inf")
            )
            dists.append(min_dist)

        for b2, d2 in intervals2:
            min_dist = (
                min(
                    (abs(b1 - b2) ** p + abs(d1 - d2) ** p) ** (1 / p)
                    for b1, d1 in intervals1
                )
                if intervals1
                else float("inf")
            )
            dists.append(min_dist)

        return (sum(d**p for d in dists) / len(dists)) ** (1 / p)

    def bottleneck_distance(
        self,
        barcode1: PersistenceBarcode,
        barcode2: PersistenceBarcode,
        delta: float = 0.01,
    ) -> float:
        """
        Compute bottleneck distance between barcodes.

        Args:
            barcode1: First barcode
            barcode2: Second barcode
            delta: Matching precision

        Returns:
            Distance value
        """
        intervals1 = [
            (i.birth, i.death)
            for i in barcode1.intervals
            if i.persistence != float("inf")
        ]
        intervals2 = [
            (i.birth, i.death)
            for i in barcode2.intervals
            if i.persistence != float("inf")
        ]

        if not intervals1:
            intervals1 = [(0, 0)]
        if not intervals2:
            intervals2 = [(0, 0)]

        max_val = max(max(abs(b1), abs(d1)) for b1, d1 in intervals1 + intervals2)

        best_epsilon = float("inf")

        for epsilon in np.arange(0, max_val, delta):
            matched = self._check_matching(intervals1, intervals2, epsilon)
            if matched:
                best_epsilon = epsilon
                break

        return best_epsilon

    def _check_matching(
        self,
        intervals1: List[Tuple[float, float]],
        intervals2: List[Tuple[float, float]],
        epsilon: float,
    ) -> bool:
        """Check if epsilon-matching exists."""
        used2 = set()

        for b1, d1 in intervals1:
            found = False
            for i, (b2, d2) in enumerate(intervals2):
                if i in used2:
                    continue
                if abs(b1 - b2) <= epsilon and abs(d1 - d2) <= epsilon:
                    used2.add(i)
                    found = True
                    break
            if not found:
                return False

        return True


def diagram_to_barcode(
    diagram,
    max_intervals: Optional[int] = None,
) -> PersistenceBarcode:
    """
    Convert persistence diagram to barcode.

    Args:
        diagram: PersistenceDiagram object
        max_intervals: Maximum number of intervals to keep

    Returns:
        PersistenceBarcode
    """
    intervals = []

    for pair in diagram.pairs:
        interval = BarcodeInterval(
            birth=pair.birth,
            death=pair.death,
            dimension=pair.dimension,
            multiplicity=pair.multiplicity,
        )
        intervals.append(interval)

    intervals = sorted(intervals, key=lambda x: x.persistence, reverse=True)

    if max_intervals is not None:
        intervals = intervals[:max_intervals]

    return PersistenceBarcode(diagram.dimension, intervals)
