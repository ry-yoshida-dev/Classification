from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np

from ..dataset import SupervisedDataset
from ..metrics import ClassScoreKind, ClassScores
from ..processor import ClassificationProcessor
from ..types import FloatArray, LabelArray, NumericArray
from .protocols.margin import MarginClassifierLike


@dataclass(eq=False)
class MarginClassificationProcessor(ClassificationProcessor):
    """
    Wraps classifiers that produce decision margins instead of probabilities.

    It deliberately offers no ``predict_proba``, so asking these classifiers for
    probabilities is rejected by the type checker instead of at runtime.

    Attributes
    ----------
    estimator : MarginClassifierLike
        The underlying estimator (structural type).
    """

    estimator: MarginClassifierLike

    @property
    @override
    def score_kind(self) -> ClassScoreKind:
        """
        Get the meaning of the per-class scores of this processor.

        Returns
        -------
        ClassScoreKind
            Always ClassScoreKind.MARGIN.
        """
        return ClassScoreKind.MARGIN

    def decision_function(self, X: NumericArray) -> ClassScores:
        """
        Compute one-vs-rest decision margins.

        Binary classifiers yield one margin per sample; it is expanded here into
        one column per class so that all scores share the same shape.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        ClassScores
            The margins with shape (n_samples, n_classes).

        Raises
        ------
        NotFittedError
            If fit has not completed successfully yet.
        """
        return self._compute_class_scores(X)

    @override
    def _fit_estimator(self, dataset: SupervisedDataset) -> None:
        _ = self.estimator.fit(dataset.X, dataset.y)

    @override
    def _predict_labels(self, X: NumericArray) -> LabelArray:
        return self.estimator.predict(X)

    @override
    def _compute_class_scores(self, X: NumericArray) -> ClassScores:
        classes: LabelArray = self.classes
        margins: FloatArray = self.estimator.decision_function(X)
        if margins.ndim == 1:
            margins = np.column_stack([-margins, margins])
        return ClassScores(classes=classes, values=margins, kind=self.score_kind)
