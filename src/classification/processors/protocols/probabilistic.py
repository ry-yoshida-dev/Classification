"""Structural type for classifiers used by ProbabilisticClassificationProcessor."""

from __future__ import annotations

from typing import Protocol

from ...types import FloatArray, LabelArray, NumericArray


class ProbabilisticClassifierLike(Protocol):
    """
    Structural type for classifiers that estimate class probabilities.

    Probability columns must follow the sorted order of the class labels, as
    they do for all scikit-learn classifiers.
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

    def predict_proba(self, X: NumericArray) -> FloatArray:
        """
        Estimate class membership probabilities.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        FloatArray
            The probabilities with shape (n_samples, n_classes).
        """
        ...
