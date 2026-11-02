"""Structural type for classifiers used by MarginClassificationProcessor."""

from __future__ import annotations

from typing import Protocol

from ...types import FloatArray, LabelArray, NumericArray


class MarginClassifierLike(Protocol):
    """
    Structural type for classifiers that produce decision margins.

    Margins have shape (n_samples,) for two classes, where positive values favor
    the second class, and (n_samples, n_classes) otherwise, with columns in the
    sorted order of the class labels.
    """

    def fit(self, X: NumericArray, y: LabelArray) -> object:
        """
        Fit the classifier.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        y : LabelArray
            The class labels.

        Returns
        -------
        object
            The return value is ignored.
        """
        ...

    def predict(self, X: NumericArray) -> LabelArray:
        """
        Predict class labels.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        LabelArray
            The predicted labels with shape (n_samples,).
        """
        ...

    def decision_function(self, X: NumericArray) -> FloatArray:
        """
        Compute decision margins.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        FloatArray
            The margins, shaped as described on this protocol.
        """
        ...
