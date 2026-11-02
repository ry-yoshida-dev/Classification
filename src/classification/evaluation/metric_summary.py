from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import FloatArray


@dataclass(frozen=True)
class MetricSummary:
    """
    Aggregate of one metric over evaluation folds.

    Attributes
    ----------
    mean : float
        Mean value.
    std : float | None
        Sample standard deviation; None with fewer than two folds, where the
        variability is unknown rather than zero.
    minimum : float
        Smallest value.
    maximum : float
        Largest value.
    num_folds : int
        Number of folds in which the metric is defined.
    """

    mean: float
    std: float | None
    minimum: float
    maximum: float
    num_folds: int

    @classmethod
    def from_values(cls, values: list[float]) -> MetricSummary:
        """
        Summarize per-fold metric values.

        Parameters
        ----------
        values : list[float]
            The value of each fold.

        Returns
        -------
        MetricSummary
            The summary.

        Raises
        ------
        ValueError
            If values is empty.
        """
        if not values:
            raise ValueError("values must not be empty")
        array: FloatArray = np.array(values, dtype=np.float64)
        return cls(
            mean=float(np.mean(array)),
            std=float(np.std(array, ddof=1)) if array.size > 1 else None,
            minimum=float(np.min(array)),
            maximum=float(np.max(array)),
            num_folds=int(array.size),
        )

    def format(self) -> str:
        """
        Format the mean and standard deviation for display.

        Returns
        -------
        str
            The formatted summary.
        """
        if self.std is None:
            return f"{self.mean:.4f}"
        return f"{self.mean:.4f} ± {self.std:.4f}"
