"""
SHAP-based Feature Importance Selector for fishstick

Feature importance using SHAP values.
"""

from typing import Optional, Union
import numpy as np
import torch

from . import SupervisedSelector, SelectionResult


class SHAPImportanceSelector(SupervisedSelector):
    """
    SHAP-based feature importance selector.

    Uses SHAP (SHapley Additive exPlanations) values to compute
    feature importance scores.

    Args:
        estimator: Base estimator (must be compatible with SHAP)
        n_features_to_select: Number of features to select
        n_samples: Number of samples to use for SHAP (None = all)
        random_state: Random seed

    Example:
        >>> selector = SHAPImportanceSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: Optional = None,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_samples: Optional[int] = 100,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_features_to_select=n_features_to_select)
        self.estimator = estimator
        self.n_samples = n_samples
        self.random_state = random_state
        self.shap_values_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "SHAPImportanceSelector":
        """
        Compute SHAP importance scores.

        Args:
            X: Input features (n_samples, n_features)
            y: Target labels

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier

            self.estimator = RandomForestClassifier(n_estimators=50, random_state=42)

        n_samples = X_np.shape[0]

        if self.n_samples is not None and self.n_samples < n_samples:
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(n_samples, self.n_samples, replace=False)
            X_sample = X_np[indices]
            y_sample = y_np[indices]
        else:
            X_sample = X_np
            y_sample = y_np

        self.estimator.fit(X_sample, y_sample)

        try:
            import shap

            if hasattr(self.estimator, "predict_proba"):
                explainer = shap.TreeExplainer(self.estimator)
                shap_values = explainer.shap_values(X_sample)

                if isinstance(shap_values, list):
                    shap_values = np.abs(shap_values[0])
                else:
                    shap_values = np.abs(shap_values)

            else:
                background = shap.kmeans(X_sample, 50)
                explainer = shap.KernelExplainer(self.estimator.predict, background)
                shap_values = explainer.shap_values(X_sample[: min(100, len(X_sample))])
                shap_values = np.abs(shap_values)

            self.shap_values_ = shap_values
            self.scores_ = np.mean(shap_values, axis=0)

        except ImportError:
            self.scores_ = np.ones(self.n_features_in_)

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]

        self.selected_features_ = indices

        return self


class KernelSHAPSelector(SHAPImportanceSelector):
    """
    Kernel SHAP selector for model-agnostic feature importance.

    Uses Kernel SHAP which works with any model.

    Args:
        estimator: Base estimator
        n_features_to_select: Number of features to select
        n_samples: Number of samples to use
        nsamples: Number of SHAP evaluations
        random_state: Random seed
    """

    def __init__(
        self,
        estimator: Optional = None,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_samples: int = 100,
        nsamples: int = 100,
        random_state: Optional[int] = 42,
    ):
        super().__init__(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            n_samples=n_samples,
            random_state=random_state,
        )
        self.nsamples = nsamples

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "KernelSHAPSelector":
        """Compute Kernel SHAP importance."""
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        if self.estimator is None:
            from sklearn.linear_model import LogisticRegression

            self.estimator = LogisticRegression(random_state=42, max_iter=1000)

        n_samples = min(
            self.n_samples if self.n_samples else X_np.shape[0], X_np.shape[0]
        )

        if self.n_samples is not None and self.n_samples < X_np.shape[0]:
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(X_np.shape[0], n_samples, replace=False)
            X_sample = X_np[indices]
        else:
            X_sample = X_np[:n_samples]

        self.estimator.fit(X_sample, y_np[:n_samples])

        try:
            import shap

            if hasattr(self.estimator, "predict_proba"):

                def predict_fn(x):
                    return self.estimator.predict_proba(x)[:, 1]
            else:
                predict_fn = self.estimator.predict

            background = shap.kmeans(X_sample, min(50, len(X_sample)))
            explainer = shap.KernelExplainer(predict_fn, background)

            X_eval = X_sample[: min(50, len(X_sample))]
            shap_values = explainer.shap_values(X_eval, nsamples=self.nsamples)

            self.shap_values_ = shap_values
            self.scores_ = np.mean(np.abs(shap_values), axis=0)

        except ImportError:
            self.scores_ = np.ones(self.n_features_in_)

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]

        self.selected_features_ = indices

        return self


def shap_importance(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    estimator: Optional = None,
    n_features: Optional[int] = None,
) -> SelectionResult:
    """
    Functional interface for SHAP-based feature selection.

    Args:
        X: Input features
        y: Target labels
        estimator: Base estimator
        n_features: Number of features to select

    Returns:
        SelectionResult
    """
    selector = SHAPImportanceSelector(
        estimator=estimator,
        n_features_to_select=n_features,
    )
    selector.fit(X, y)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method="shap_importance",
    )
