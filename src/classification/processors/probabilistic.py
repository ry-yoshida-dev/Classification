from __future__ import annotations

from dataclasses import dataclass
from typing import override

from ..dataset import SupervisedDataset
from ..metrics import ClassScoreKind, ClassScores
from ..processor import ClassificationProcessor
from ..types import LabelArray, NumericArray
from .protocols.probabilistic import ProbabilisticClassifierLike


@dataclass(eq=False)
class ProbabilisticClassificationProcessor(ClassificationProcessor):
    """
    Wraps classifiers that estimate class membership probabilities.

    Attributes
    ----------
    estimator : ProbabilisticClassifierLike
        The underlying estimator (structural type).
    """

    estimator: ProbabilisticClassifierLike

    @property
    @override
    def score_kind(self) -> ClassScoreKind:
        """
        Get the meaning of the per-class scores of this processor.

        Returns
        -------
        ClassScoreKind
            Always ClassScoreKind.PROBABILITY.
        """
        return ClassScoreKind.PROBABILITY

    def predict_proba(self, X: NumericArray) -> ClassScores:
        """
        Estimate class membership probabilities.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        ClassScores
            The probabilities with shape (n_samples, n_classes).

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
        return ClassScores(
            classes=self.classes,
            values=self.estimator.predict_proba(X),
            kind=self.score_kind,
        )
