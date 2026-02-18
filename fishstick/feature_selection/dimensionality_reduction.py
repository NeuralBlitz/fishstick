"""
Dimensionality Reduction Feature Selectors for fishstick

PCA, LDA, and ICA-based feature extraction.
"""

from typing import Optional, Union, Tuple
import numpy as np
import torch
from sklearn.decomposition import PCA as SklearnPCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import BaseSelector, SupervisedSelector, UnsupervisedSelector, SelectionResult


class PCASelector(BaseSelector):
    """
    PCA-based feature selector.

    Uses Principal Component Analysis to project features to
    a lower-dimensional subspace while preserving variance.

    Args:
        n_components: Number of components (int or float 0-1 for variance)

    Example:
        >>> selector = PCASelector(n_components=10)
        >>> X_transformed = selector.fit_transform(X)
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
    ):
        super().__init__(n_features_to_select=n_components)
        self.n_components = n_components
        self.pca_: Optional[SklearnPCA] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> "PCASelector":
        """
        Fit PCA.

        Args:
            X: Input features (n_samples, n_features)
            y: Ignored (unsupervised)

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)

        self.n_features_in_ = X_np.shape[1]

        if self.n_components is None:
            n_comp = min(X_np.shape)
        elif isinstance(self.n_components, float):
            n_comp = self.n_components
        else:
            n_comp = self.n_components

        self.pca_ = SklearnPCA(n_components=n_comp)
        self.pca_.fit(X_np)

        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_

        if isinstance(self.n_components, int):
            n_select = self.n_components
        else:
            cumvar = np.cumsum(self.explained_variance_ratio_)
            n_select = (
                np.searchsorted(cumvar, self.n_components) + 1
                if self.n_components < 1.0
                else self.n_features_in_
            )

        self.selected_features_ = np.arange(min(n_select, self.n_features_in_))
        self.scores_ = self.explained_variance_ratio_

        return self

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform to principal component space."""
        if self.pca_ is None:
            raise ValueError("Selector has not been fitted.")

        X_np, is_torch = self._to_numpy(X)

        result = self.pca_.transform(X_np)

        return self._to_torch(result, is_torch)

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Inverse transform from PCA space."""
        if self.pca_ is None:
            raise ValueError("Selector has not been fitted.")

        X_np, is_torch = self._to_numpy(X)

        result = self.pca_.inverse_transform(X_np)

        return self._to_torch(result, is_torch)


class LDASelector(SupervisedSelector):
    """
    Linear Discriminant Analysis feature selector.

    Uses LDA to project features to maximize class separability.
    Supervised method that requires target labels.

    Args:
        n_components: Maximum = n_classes - 1

    Example:
        >>> selector = LDASelector(n_components=2)
        >>> X_transformed = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
    ):
        super().__init__(n_features_to_select=n_components)
        self.n_components = n_components
        self.lda_: Optional[LinearDiscriminantAnalysis] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "LDASelector":
        """
        Fit LDA.

        Args:
            X: Input features
            y: Target labels

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        n_classes = len(np.unique(y_np))
        max_components = min(n_classes - 1, X_np.shape[1])

        if self.n_components is None:
            n_comp = max_components
        else:
            n_comp = min(self.n_components, max_components)

        self.lda_ = LinearDiscriminantAnalysis(n_components=n_comp)
        self.lda_.fit(X_np, y_np)

        self.explained_variance_ratio_ = self.lda_.explained_variance_ratio_

        self.selected_features_ = np.arange(n_comp)
        self.scores_ = self.explained_variance_ratio_

        return self

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform to LDA space."""
        if self.lda_ is None:
            raise ValueError("Selector has not been fitted.")

        X_np, is_torch = self._to_numpy(X)

        result = self.lda_.transform(X_np)

        return self._to_torch(result, is_torch)

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)


class ICASelector(UnsupervisedSelector):
    """
    Independent Component Analysis feature selector.

    Uses ICA to find independent components in the data.

    Args:
        n_components: Number of independent components
        max_iter: Maximum iterations for convergence
        tol: Tolerance for convergence
        random_state: Random seed

    Example:
        >>> selector = ICASelector(n_components=10)
        >>> X_transformed = selector.fit_transform(X)
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        max_iter: int = 1000,
        tol: float = 0.01,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_features_to_select=n_components)
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.ica_: Optional[FastICA] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> "ICASelector":
        """
        Fit ICA.

        Args:
            X: Input features
            y: Ignored

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)

        self.n_features_in_ = X_np.shape[1]

        if self.n_components is None:
            n_comp = min(X_np.shape)
        else:
            n_comp = self.n_components

        self.ica_ = FastICA(
            n_components=n_comp,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        self.ica_.fit(X_np)

        self.selected_features_ = np.arange(n_comp)
        self.scores_ = np.ones(n_comp)

        return self

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform to independent component space."""
        if self.ica_ is None:
            raise ValueError("Selector has not been fitted.")

        X_np, is_torch = self._to_numpy(X)

        result = self.ica_.transform(X_np)

        return self._to_torch(result, is_torch)

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)


class KernelPCASelector(BaseSelector):
    """
    Kernel PCA feature selector.

    Uses kernel PCA for non-linear dimensionality reduction.

    Args:
        n_components: Number of components
        kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
        gamma: Kernel coefficient for rbf/poly
        degree: Degree for poly kernel

    Example:
        >>> selector = KernelPCASelector(n_components=10, kernel='rbf')
        >>> X_transformed = selector.fit_transform(X)
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
    ):
        super().__init__(n_features_to_select=n_components)
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> "KernelPCASelector":
        """Fit Kernel PCA."""
        from sklearn.decomposition import KernelPCA

        X_np, is_torch = self._to_numpy(X)

        self.n_features_in_ = X_np.shape[1]

        n_comp = self.n_components or min(X_np.shape)

        self.kpca_ = KernelPCA(
            n_components=n_comp,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
        )

        self.kpca_.fit(X_np)

        self.selected_features_ = np.arange(n_comp)
        self.scores_ = np.ones(n_comp)

        return self

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform using Kernel PCA."""
        X_np, is_torch = self._to_numpy(X)

        result = self.kpca_.transform(X_np)

        return self._to_torch(result, is_torch)

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)


def pca_selector(
    X: Union[np.ndarray, torch.Tensor],
    n_components: Optional[int] = None,
) -> SelectionResult:
    """
    Functional interface for PCA feature selection.

    Args:
        X: Input features
        n_components: Number of components

    Returns:
        SelectionResult
    """
    selector = PCASelector(n_components=n_components)
    selector.fit(X)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method="pca",
        metadata={"explained_variance_ratio": selector.explained_variance_ratio_},
    )
