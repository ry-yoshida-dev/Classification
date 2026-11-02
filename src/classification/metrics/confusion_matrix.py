from __future__ import annotations

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from sklearn.metrics import confusion_matrix

from ..types import (
    BoolArray,
    ClassLabel,
    CountArray,
    FloatArray,
    IndexArray,
    LabelArray,
)


@dataclass(frozen=True, eq=False)
class ConfusionMatrix:
    """
    Counts of true classes (rows) against predicted classes (columns).

    Attributes
    ----------
    classes : LabelArray
        The read-only sorted class labels of the rows and columns.
    counts : CountArray
        The read-only counts with shape (n_classes, n_classes).

    Raises
    ------
    ValueError
        If the counts do not form a square matrix over the classes.
    """

    classes: LabelArray
    counts: CountArray

    def __post_init__(self) -> None:
        classes: LabelArray = np.array(self.classes)
        counts: CountArray = np.array(self.counts, dtype=np.int64)
        expected_shape: tuple[int, int] = (self.classes.size, self.classes.size)
        if counts.shape != expected_shape:
            raise ValueError(
                f"counts must have shape {expected_shape}, got {counts.shape}"
            )
        classes.setflags(write=False)
        counts.setflags(write=False)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "counts", counts)

    @classmethod
    def from_predictions(
        cls,
        y_true: LabelArray,
        y_pred: LabelArray,
        classes: LabelArray,
    ) -> ConfusionMatrix:
        """
        Count true and predicted labels.

        Parameters
        ----------
        y_true : LabelArray
            The true labels.
        y_pred : LabelArray
            The predicted labels.
        classes : LabelArray
            The sorted class labels of the rows and columns.

        Returns
        -------
        ConfusionMatrix
            The confusion matrix.
        """
        return cls(
            classes=classes,
            counts=confusion_matrix(y_true, y_pred, labels=classes),
        )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConfusionMatrix):
            return NotImplemented
        return bool(
            np.array_equal(self.classes, other.classes)
            and np.array_equal(self.counts, other.counts)
        )

    @override
    def __hash__(self) -> int:
        labels: list[ClassLabel] = cast(list[ClassLabel], self.classes.tolist())
        return hash((tuple(labels), self.counts.tobytes()))

    @property
    def row_normalized(self) -> FloatArray:
        """
        Get the fraction of each true class assigned to each predicted class.

        Returns
        -------
        FloatArray
            The fractions with shape (n_classes, n_classes); rows of classes
            without samples are all zero.
        """
        totals: CountArray = self.counts.sum(axis=1, keepdims=True)
        return np.divide(
            self.counts,
            totals,
            out=np.zeros(self.counts.shape, dtype=np.float64),
            where=totals > 0,
        )

    def get_count(self, true_label: ClassLabel, predicted_label: ClassLabel) -> int:
        """
        Get the number of samples of a true class assigned to a predicted class.

        Parameters
        ----------
        true_label : ClassLabel
            The true class label.
        predicted_label : ClassLabel
            The predicted class label.

        Returns
        -------
        int
            The number of samples.

        Raises
        ------
        KeyError
            If either label is not one of the classes.
        """
        return self.counts.item(
            self._position(true_label), self._position(predicted_label)
        )

    def combine(self, other: ConfusionMatrix) -> ConfusionMatrix:
        """
        Add another confusion matrix, aligning on the union of their classes.

        Parameters
        ----------
        other : ConfusionMatrix
            The confusion matrix to add.

        Returns
        -------
        ConfusionMatrix
            The combined confusion matrix.
        """
        classes: LabelArray = np.union1d(self.classes, other.classes)
        counts: CountArray = np.zeros((classes.size, classes.size), dtype=np.int64)
        for matrix in (self, other):
            positions: IndexArray = np.searchsorted(classes, matrix.classes)
            counts[np.ix_(positions, positions)] += matrix.counts
        return ConfusionMatrix(classes=classes, counts=counts)

    def _position(self, label: ClassLabel) -> int:
        is_match: BoolArray = cast(BoolArray, self.classes == label)
        positions: IndexArray = np.flatnonzero(is_match)
        if positions.size == 0:
            raise KeyError(label)
        return positions.item(0)
