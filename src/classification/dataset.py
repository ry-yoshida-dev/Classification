from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import IndexArray, LabelArray, NumericArray


@dataclass(frozen=True, eq=False)
class SupervisedDataset:
    """
    Validated pair of feature matrix and class labels.

    Attributes
    ----------
    X : NumericArray
        Feature matrix with shape (n_samples, n_features).
    y : LabelArray
        Integer, boolean, or string class labels with shape (n_samples,).

    Raises
    ------
    TypeError
        If the features are not numeric or the labels are not discrete.
    ValueError
        If the shapes are invalid or inconsistent.
    """

    X: NumericArray
    y: LabelArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "X", np.asarray(self.X))
        object.__setattr__(self, "y", np.asarray(self.y))
        self._validate_features()
        self._validate_labels()

    @property
    def num_samples(self) -> int:
        """
        Get the number of samples.

        Returns
        -------
        int
            The number of samples.
        """
        return len(self.X)

    @property
    def classes(self) -> LabelArray:
        """
        Get the sorted unique class labels.

        Returns
        -------
        LabelArray
            The sorted unique class labels.
        """
        return np.unique(self.y)

    def subset(self, indices: IndexArray) -> SupervisedDataset:
        """
        Select the samples at the given indices.

        Parameters
        ----------
        indices : IndexArray
            The sample indices to select.

        Returns
        -------
        SupervisedDataset
            The selected samples.
        """
        return SupervisedDataset(X=self.X[indices], y=self.y[indices])

    def _validate_features(self) -> None:
        if not np.issubdtype(self.X.dtype, np.integer) and not np.issubdtype(
            self.X.dtype, np.floating
        ):
            raise TypeError(f"X must be numeric, got dtype {self.X.dtype}")
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {self.X.ndim} dimensions")
        if self.X.shape[0] == 0 or self.X.shape[1] == 0:
            raise ValueError(f"X must not be empty, got shape {self.X.shape}")

    def _validate_labels(self) -> None:
        is_discrete: bool = any(
            np.issubdtype(self.y.dtype, label_type)
            for label_type in (np.integer, np.bool_, np.str_)
        )
        if not is_discrete:
            raise TypeError(
                f"y must contain integer, boolean, or string labels, got dtype {self.y.dtype}"
            )
        if self.y.ndim != 1:
            raise ValueError(f"y must be 1-D, got {self.y.ndim} dimensions")
        if self.y.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"X and y must have the same length, got {self.X.shape[0]} and {self.y.shape[0]}"
            )
