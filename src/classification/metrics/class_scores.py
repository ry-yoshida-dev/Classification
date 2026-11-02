from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from ..types import BoolArray, ClassLabel, FloatArray, IndexArray, LabelArray
from .class_score_kind import ClassScoreKind


@dataclass(frozen=True, eq=False)
class ClassScores:
    """
    Per-class scores whose columns are explicitly tied to class labels.

    Attributes
    ----------
    classes : LabelArray
        The read-only class labels, one per score column.
    values : FloatArray
        The read-only scores with shape (n_samples, n_classes).
    kind : ClassScoreKind
        The meaning of the scores.

    Raises
    ------
    ValueError
        If the score columns do not match the classes.
    """

    classes: LabelArray
    values: FloatArray
    kind: ClassScoreKind

    def __post_init__(self) -> None:
        classes: LabelArray = np.array(self.classes)
        values: FloatArray = np.array(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.classes.size:
            raise ValueError(
                f"values must have shape (n_samples, {self.classes.size}), got {values.shape}"
            )
        classes.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "values", values)

    def get_column(self, label: ClassLabel) -> FloatArray:
        """
        Get the scores of one class.

        Parameters
        ----------
        label : ClassLabel
            The class label.

        Returns
        -------
        FloatArray
            The scores with shape (n_samples,).

        Raises
        ------
        KeyError
            If the label is not one of the classes.
        """
        is_match: BoolArray = cast(BoolArray, self.classes == label)
        positions: IndexArray = np.flatnonzero(is_match)
        if positions.size == 0:
            raise KeyError(label)
        return self.values[:, positions.item(0)]
