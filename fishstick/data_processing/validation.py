"""
Data Validation Module for fishstick

Provides comprehensive data validation tools for ensuring data quality
including schema validation, range checking, statistical validation,
and duplicate detection.

Features:
- Schema-based validation
- Value range validation
- Statistical property checks
- Duplicate detection
- Detailed validation reports
- Auto-validating datasets
"""

from __future__ import annotations

from typing import (
    Optional,
    Callable,
    List,
    Union,
    Dict,
    Any,
    Tuple,
    Sequence,
    TypeVar,
    Generic,
    Protocol,
    runtime_checkable,
)
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
import hashlib
import numpy as np
import torch
from torch import Tensor


T = TypeVar("T")
ArrayLike = Union[np.ndarray, Tensor]


class ValidationLevel(Enum):
    """Validation severity levels."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Single validation issue."""

    level: ValidationLevel
    message: str
    field: Optional[str] = None
    value: Any = None
    index: Optional[int] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None

    def add_issue(
        self,
        level: ValidationLevel,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
    ) -> None:
        """Add a validation issue."""
        self.issues.append(
            ValidationIssue(level=level, message=message, field=field, value=value)
        )
        if level == ValidationLevel.ERROR or level == ValidationLevel.CRITICAL:
            self.valid = False

    def summary(self) -> str:
        """Get summary string."""
        counts = {level: 0 for level in ValidationLevel}
        for issue in self.issues:
            counts[issue.level] += 1

        return (
            f"Validation {'PASSED' if self.valid else 'FAILED'}: "
            f"{counts[ValidationLevel.CRITICAL]} critical, "
            f"{counts[ValidationLevel.ERROR]} errors, "
            f"{counts[ValidationLevel.WARNING]} warnings"
        )

    def __str__(self) -> str:
        lines = [self.summary()]
        for issue in self.issues:
            field_str = f" [{issue.field}]" if issue.field else ""
            lines.append(f"  {issue.level.value.upper()}{field_str}: {issue.message}")
        return "\n".join(lines)


class SchemaValidator:
    """
    Schema-based data validation.

    Validates data against a defined schema with type checking,
    shape constraints, and custom validators.
    """

    @dataclass
    class FieldSchema:
        """Schema for a single field."""

        dtype: Optional[Union[type, Tuple[type, ...]]] = None
        shape: Optional[Tuple[Optional[int], ...]] = None
        min_value: Optional[float] = None
        max_value: Optional[float] = None
        nullable: bool = True
        choices: Optional[Sequence] = None
        custom: Optional[Callable[[Any], bool]] = None

    def __init__(self, schema: Dict[str, FieldSchema]):
        """
        Args:
            schema: Dict mapping field names to FieldSchema
        """
        self.schema = schema

    def validate(self, data: Dict[str, Any]) -> ValidationReport:
        """Validate data against schema."""
        report = ValidationReport(valid=True)

        for field_name, field_schema in self.schema.items():
            if field_name not in data:
                if not field_schema.nullable:
                    report.add_issue(
                        ValidationLevel.ERROR,
                        f"Missing required field: {field_name}",
                        field=field_name,
                    )
                continue

            value = data[field_name]
            self._validate_field(field_name, value, field_schema, report)

        return report

    def _validate_field(
        self,
        name: str,
        value: Any,
        schema: FieldSchema,
        report: ValidationReport,
    ) -> None:
        if value is None:
            if not schema.nullable:
                report.add_issue(
                    ValidationLevel.ERROR, f"Field cannot be null", field=name
                )
            return

        if schema.dtype is not None:
            if not isinstance(value, schema.dtype):
                report.add_issue(
                    ValidationLevel.ERROR,
                    f"Invalid type: expected {schema.dtype}, got {type(value)}",
                    field=name,
                    value=type(value).__name__,
                )
                return

        if isinstance(value, (int, float)):
            if schema.min_value is not None and value < schema.min_value:
                report.add_issue(
                    ValidationLevel.ERROR,
                    f"Value {value} below minimum {schema.min_value}",
                    field=name,
                    value=value,
                )
            if schema.max_value is not None and value > schema.max_value:
                report.add_issue(
                    ValidationLevel.ERROR,
                    f"Value {value} above maximum {schema.max_value}",
                    field=name,
                    value=value,
                )

        if schema.shape is not None:
            if hasattr(value, "shape"):
                actual_shape = value.shape
                for i, (exp, act) in enumerate(zip(schema.shape, actual_shape)):
                    if exp is not None and exp != act:
                        report.add_issue(
                            ValidationLevel.ERROR,
                            f"Shape mismatch at dim {i}: expected {exp}, got {act}",
                            field=name,
                            value=actual_shape,
                        )

        if schema.choices is not None:
            if value not in schema.choices:
                report.add_issue(
                    ValidationLevel.ERROR,
                    f"Value not in allowed choices: {schema.choices}",
                    field=name,
                    value=value,
                )

        if schema.custom is not None:
            try:
                if not schema.custom(value):
                    report.add_issue(
                        ValidationLevel.ERROR,
                        "Custom validation failed",
                        field=name,
                        value=value,
                    )
            except Exception as e:
                report.add_issue(
                    ValidationLevel.ERROR,
                    f"Custom validator error: {e}",
                    field=name,
                    value=value,
                )


class RangeValidator:
    """
    Value range validation.

    Checks that values fall within specified ranges.
    """

    def __init__(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
    ):
        """
        Args:
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_nan: Whether NaN is allowed
            allow_inf: Whether Inf is allowed
        """
        self.min_val = min_val
        self.max_val = max_val
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf

    def validate(self, data: ArrayLike) -> ValidationReport:
        """Validate array against range constraints."""
        report = ValidationReport(valid=True)
        data_arr = self._to_numpy(data)

        if not self.allow_nan:
            nan_count = np.isnan(data_arr).sum()
            if nan_count > 0:
                report.add_issue(
                    ValidationLevel.ERROR,
                    f"Found {nan_count} NaN values",
                    value=nan_count,
                )

        if not self.allow_inf:
            inf_count = np.isinf(data_arr).sum()
            if inf_count > 0:
                report.add_issue(
                    ValidationLevel.ERROR,
                    f"Found {inf_count} infinite values",
                    value=inf_count,
                )

        if self.min_val is not None:
            below_count = (data_arr < self.min_val).sum()
            if below_count > 0:
                report.add_issue(
                    ValidationLevel.WARNING,
                    f"{below_count} values below minimum {self.min_val}",
                    value=below_count,
                )

        if self.max_val is not None:
            above_count = (data_arr > self.max_val).sum()
            if above_count > 0:
                report.add_issue(
                    ValidationLevel.WARNING,
                    f"{above_count} values above maximum {self.max_val}",
                    value=above_count,
                )

        return report

    def _to_numpy(self, data: ArrayLike) -> np.ndarray:
        if isinstance(data, Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)


class StatisticalValidator:
    """
    Statistical property validation.

    Checks statistical properties like distribution, variance, etc.
    """

    def __init__(
        self,
        expected_mean: Optional[float] = None,
        expected_std: Optional[float] = None,
        expected_distribution: Optional[str] = None,
        min_variance: float = 0.0,
        check_normality: bool = False,
    ):
        """
        Args:
            expected_mean: Expected mean
            expected_std: Expected std
            expected_distribution: Expected distribution name
            min_variance: Minimum variance threshold
            check_normality: Whether to test normality
        """
        self.expected_mean = expected_mean
        self.expected_std = expected_std
        self.expected_distribution = expected_distribution
        self.min_variance = min_variance
        self.check_normality = check_normality

    def validate(self, data: ArrayLike) -> ValidationReport:
        """Validate statistical properties."""
        report = ValidationReport(valid=True)
        data_arr = self._to_numpy(data).flatten()

        mean = data_arr.mean()
        std = data_arr.std()
        variance = data_arr.var()

        report.stats = {
            "mean": float(mean),
            "std": float(std),
            "variance": float(variance),
            "min": float(data_arr.min()),
            "max": float(data_arr.max()),
            "n_samples": len(data_arr),
        }

        if self.expected_mean is not None:
            if abs(mean - self.expected_mean) > 0.1 * abs(self.expected_mean):
                report.add_issue(
                    ValidationLevel.WARNING,
                    f"Mean {mean:.4f} differs from expected {self.expected_mean:.4f}",
                    value=mean,
                )

        if self.expected_std is not None:
            if abs(std - self.expected_std) > 0.1 * abs(self.expected_std):
                report.add_issue(
                    ValidationLevel.WARNING,
                    f"Std {std:.4f} differs from expected {self.expected_std:.4f}",
                    value=std,
                )

        if variance < self.min_variance:
            report.add_issue(
                ValidationLevel.WARNING,
                f"Variance {variance:.6f} below minimum {self.min_variance}",
                value=variance,
            )

        if self.check_normality:
            from scipy import stats

            _, p_value = stats.normaltest(data_arr)
            report.stats["normality_p_value"] = p_value
            if p_value < 0.05:
                report.add_issue(
                    ValidationLevel.WARNING,
                    f"Data may not be normally distributed (p={p_value:.4f})",
                    value=p_value,
                )

        return report

    def _to_numpy(self, data: ArrayLike) -> np.ndarray:
        if isinstance(data, Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)


class DuplicateValidator:
    """
    Duplicate detection validator.

    Finds and reports duplicate samples.
    """

    def __init__(
        self,
        subset: Optional[List[str]] = None,
        keep: str = "first",
        tolerance: float = 1e-6,
    ):
        """
        Args:
            subset: Columns to consider for duplicates
            keep: Which duplicate to keep ('first', 'last', False)
            tolerance: Tolerance for float comparison
        """
        self.subset = subset
        self.keep = keep
        self.tolerance = tolerance

    def validate(self, data: ArrayLike) -> ValidationReport:
        """Detect duplicates in data."""
        report = ValidationReport(valid=True)
        data_arr = self._to_numpy(data)

        if data_arr.ndim == 1:
            data_arr = data_arr.reshape(-1, 1)

        rounded = np.round(data_arr, decimals=int(-np.log10(self.tolerance)))
        _, indices, counts = np.unique(
            rounded, axis=0, return_index=True, return_counts=True
        )

        duplicates = counts > 1
        n_duplicates = duplicates.sum()

        report.stats = {
            "total_samples": len(data_arr),
            "unique_samples": len(indices),
            "duplicate_count": n_duplicates,
            "duplicate_percentage": 100 * n_duplicates / len(data_arr),
        }

        if n_duplicates > 0:
            duplicate_indices = indices[duplicates]
            report.add_issue(
                ValidationLevel.WARNING,
                f"Found {n_duplicates} duplicate samples ({100 * n_duplicates / len(data_arr):.1f}%)",
                value=n_duplicates,
            )

        return report

    def find_duplicates(self, data: ArrayLike) -> List[Tuple[int, int]]:
        """Find indices of duplicate pairs."""
        data_arr = self._to_numpy(data)
        if data_arr.ndim == 1:
            data_arr = data_arr.reshape(-1, 1)

        rounded = np.round(data_arr, decimals=int(-np.log10(self.tolerance)))
        _, indices, inverse = np.unique(
            rounded, axis=0, return_index=True, return_inverse=True
        )

        seen: Dict[int, List[int]] = {}
        for i, inv in enumerate(inverse):
            if inv not in seen:
                seen[inv] = []
            seen[inv].append(i)

        duplicates = []
        for positions in seen.values():
            if len(positions) > 1:
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        duplicates.append((positions[i], positions[j]))

        return duplicates

    def _to_numpy(self, data: ArrayLike) -> np.ndarray:
        if isinstance(data, Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)


class ValidatedDataset:
    """
    Dataset with automatic validation.

    Wraps a dataset and validates samples on access.
    """

    def __init__(
        self,
        dataset: Any,
        validators: List[Any],
        validate_on: str = "access",
        cache_valid: bool = True,
    ):
        """
        Args:
            dataset: Base dataset
            validators: List of validators to apply
            validate_on: When to validate ('access', 'init')
            cache_valid: Cache validation results
        """
        self.dataset = dataset
        self.validators = validators
        self.validate_on = validate_on
        self.cache_valid = cache_valid
        self._validation_cache: Dict[int, ValidationReport] = {}

        if validate_on == "init":
            for i in range(len(dataset)):
                self._validate_sample(i)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        sample = self.dataset[idx]

        if self.validate_on == "access":
            if not self.cache_valid or idx not in self._validation_cache:
                self._validate_sample(idx)
            else:
                report = self._validation_cache[idx]
                if not report.valid:
                    raise ValueError(f"Validation failed for sample {idx}: {report}")

        return sample

    def _validate_sample(self, idx: int) -> ValidationReport:
        sample = self.dataset[idx]
        report = ValidationReport(valid=True)

        for validator in self.validators:
            if isinstance(validator, SchemaValidator):
                sample_dict = sample if isinstance(sample, dict) else {"data": sample}
                sample_report = validator.validate(sample_dict)
            elif isinstance(
                validator, (RangeValidator, StatisticalValidator, DuplicateValidator)
            ):
                sample_data = (
                    sample
                    if isinstance(sample, (np.ndarray, Tensor))
                    else sample[0]
                    if isinstance(sample, tuple)
                    else sample
                )
                sample_report = validator.validate(sample_data)
            else:
                continue

            report.issues.extend(sample_report.issues)

        if self.cache_valid:
            self._validation_cache[idx] = report

        return report


class DataIntegrityChecker:
    """
    Comprehensive data integrity checking.

    Performs multiple validation checks in one pass.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Configuration for validators
        """
        self.config = config or {}
        self.validators: List[Any] = []

        self._setup_validators()

    def _setup_validators(self) -> None:
        if "range" in self.config:
            self.validators.append(RangeValidator(**self.config["range"]))

        if "statistical" in self.config:
            self.validators.append(StatisticalValidator(**self.config["statistical"]))

        if "duplicates" in self.config:
            self.validators.append(DuplicateValidator(**self.config["duplicates"]))

        if "schema" in self.config:
            self.validators.append(SchemaValidator(self.config["schema"]))

    def validate(self, data: ArrayLike) -> ValidationReport:
        """Run all validators."""
        report = ValidationReport(valid=True)

        for validator in self.validators:
            if isinstance(validator, DuplicateValidator):
                validator_report = validator.validate(data)
            else:
                validator_report = validator.validate(data)

            report.issues.extend(validator_report.issues)
            report.stats.update(validator_report.stats)

        report.valid = not any(
            i.level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL)
            for i in report.issues
        )

        return report

    def check(self, data: ArrayLike) -> bool:
        """Quick boolean check."""
        report = self.validate(data)
        return report.valid


def validate_batch(
    batch: Union[Tensor, Dict[str, Tensor]],
    schema: Optional[Dict[str, SchemaValidator.FieldSchema]] = None,
) -> ValidationReport:
    """
    Validate a batch of data.

    Args:
        batch: Input batch (tensor or dict of tensors)
        schema: Optional schema for validation

    Returns:
        ValidationReport
    """
    report = ValidationReport(valid=True)

    if schema is not None:
        validator = SchemaValidator(schema)
        if isinstance(batch, dict):
            return validator.validate(batch)
    elif isinstance(batch, Tensor):
        range_validator = RangeValidator()
        return range_validator.validate(batch)

    return report


def validate_dataset(
    dataset: Any,
    sample_size: Optional[int] = None,
    verbose: bool = True,
) -> ValidationReport:
    """
    Validate an entire dataset.

    Args:
        dataset: Dataset to validate
        sample_size: Number of samples to check (None for all)
        verbose: Print progress

    Returns:
        Aggregated ValidationReport
    """
    report = ValidationReport(valid=True)
    n_samples = len(dataset)
    check_indices = (
        np.random.choice(
            n_samples, min(sample_size or n_samples, n_samples), replace=False
        )
        if sample_size
        else range(n_samples)
    )

    for i, idx in enumerate(check_indices):
        if verbose and (i + 1) % 1000 == 0:
            print(f"Validated {i + 1}/{len(check_indices)} samples...")

        try:
            sample = dataset[idx]
        except Exception as e:
            report.add_issue(
                ValidationLevel.ERROR,
                f"Failed to load sample {idx}: {e}",
                index=idx,
            )

    report.stats["total_samples"] = n_samples
    report.stats["checked_samples"] = len(check_indices)

    return report
