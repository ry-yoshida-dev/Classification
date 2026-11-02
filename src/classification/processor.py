from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sklearn.exceptions import NotFittedError

from .dataset import SupervisedDataset
from .method import ClassificationMethod
from .metrics import ClassScoreKind, ClassScores, MetricValues
from .types import LabelArray, NumericArray


@dataclass(eq=False)
class ClassificationProcessor(ABC):
    """
    Unified interface over a fitted-on-demand classifier.

    Subclasses provide the estimator calls; this base class validates the data
    and tracks the classes of the last successful fit. The type of per-class
    scores differs by subclass: see ProbabilisticClassificationProcessor and
    MarginClassificationProcessor.

    Attributes
    ----------
    method : ClassificationMethod
        The classification method this processor implements.
    """

    method: ClassificationMethod
    _fitted_classes: LabelArray | None = field(default=None, init=False, repr=False)

    def fit(self, X: NumericArray, y: LabelArray) -> None:
        """
        Fit the classifier.

        Parameters
        ----------
        X : NumericArray
            The feature matrix with shape (n_samples, n_features).
        y : LabelArray
            The class labels with shape (n_samples,).

        Raises
        ------
        TypeError
            If the features are not numeric or the labels are not discrete.
        ValueError
            If the shapes are invalid or y contains fewer than two classes.
        """
        self._fitted_classes = None
        dataset: SupervisedDataset = SupervisedDataset(X=X, y=y)
        classes: LabelArray = dataset.classes
        if classes.size < 2:
            raise ValueError(f"y must contain at least 2 classes, got {classes.size}")
        self._fit_estimator(dataset)
        classes.setflags(write=False)
        self._fitted_classes = classes

    @property
    @abstractmethod
    def score_kind(self) -> ClassScoreKind:
        """
        Get the meaning of the per-class scores of this processor.

        Returns
        -------
        ClassScoreKind
            The kind of the scores.
        """

    @property
    def classes(self) -> LabelArray:
        """
        Get the sorted class labels seen during the last fit.

        Returns
        -------
        LabelArray
            The read-only sorted class labels.

        Raises
        ------
        NotFittedError
            If fit has not completed successfully yet.
        """
        if self._fitted_classes is None:
            raise NotFittedError("Processor is not fitted yet. Call fit() first.")
        return self._fitted_classes

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

        Raises
        ------
        NotFittedError
            If fit has not completed successfully yet.
        """
        _ = self.classes
        return self._predict_labels(X)

    def evaluate(self, X: NumericArray, y: LabelArray) -> MetricValues:
        """
        Measure the performance of the fitted classifier on labeled data.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        y : LabelArray
            The true class labels.

        Returns
        -------
        MetricValues
            The performance metrics.

        Raises
        ------
        NotFittedError
            If fit has not completed successfully yet.
        TypeError
            If the features are not numeric or the labels are not discrete.
        ValueError
            If the shapes are invalid or y contains labels unseen during fit.
        """
        dataset: SupervisedDataset = SupervisedDataset(X=X, y=y)
        return MetricValues.compute(
            y_true=dataset.y,
            y_pred=self.predict(dataset.X),
            scores=self._compute_class_scores(dataset.X),
        )

    @abstractmethod
    def _fit_estimator(self, dataset: SupervisedDataset) -> None: ...

    @abstractmethod
    def _predict_labels(self, X: NumericArray) -> LabelArray: ...

    @abstractmethod
    def _compute_class_scores(self, X: NumericArray) -> ClassScores: ...
