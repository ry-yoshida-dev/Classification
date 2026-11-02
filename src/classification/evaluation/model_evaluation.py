from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

from ..method import ClassificationMethod
from ..metrics import ClassificationMetric, ConfusionMatrix
from .fold import FoldEvaluation
from .metric_summary import MetricSummary
from .partition import EvaluationPartition


@dataclass(frozen=True)
class ModelEvaluation:
    """
    Performance of one model configuration across all folds.

    Attributes
    ----------
    method : ClassificationMethod
        The evaluated classification method.
    folds : tuple[FoldEvaluation, ...]
        The non-empty result of each fold.

    Raises
    ------
    ValueError
        If there are no folds.
    """

    method: ClassificationMethod
    folds: tuple[FoldEvaluation, ...]

    def __post_init__(self) -> None:
        if not self.folds:
            raise ValueError("folds must not be empty")

    @property
    def confusion_matrix(self) -> ConfusionMatrix:
        """
        Get the confusion matrix summed over all test partitions.

        Returns
        -------
        ConfusionMatrix
            The combined confusion matrix.
        """
        return reduce(
            ConfusionMatrix.combine,
            [fold.test_metrics.confusion_matrix for fold in self.folds],
        )

    def is_defined(
        self,
        metric: ClassificationMetric,
        partition: EvaluationPartition = EvaluationPartition.TEST,
    ) -> bool:
        """
        Check whether a metric is defined in at least one fold.

        Parameters
        ----------
        metric : ClassificationMetric
            The metric.
        partition : EvaluationPartition
            The partition to look at.

        Returns
        -------
        bool
            True if summarize would succeed for this metric and partition.
        """
        return bool(self._collect_values(metric, partition))

    def get_defined_fold_indices(
        self,
        metric: ClassificationMetric,
        partition: EvaluationPartition = EvaluationPartition.TEST,
    ) -> tuple[int, ...]:
        """
        Get the folds in which a metric is defined.

        Means of a metric are only comparable between evaluations that are
        defined on the same folds.

        Parameters
        ----------
        metric : ClassificationMetric
            The metric.
        partition : EvaluationPartition
            The partition to look at.

        Returns
        -------
        tuple[int, ...]
            The indices of the folds in which the metric has a value.
        """
        return tuple(
            fold.fold_index
            for fold in self.folds
            if fold.get_metrics(partition).get(metric) is not None
        )

    def summarize(
        self,
        metric: ClassificationMetric,
        partition: EvaluationPartition = EvaluationPartition.TEST,
    ) -> MetricSummary:
        """
        Summarize a metric over the folds in which it is defined.

        Parameters
        ----------
        metric : ClassificationMetric
            The metric.
        partition : EvaluationPartition
            The partition to look at.

        Returns
        -------
        MetricSummary
            The summary.

        Raises
        ------
        ValueError
            If the metric is undefined in every fold.
        """
        values: list[float] = self._collect_values(metric, partition)
        if not values:
            raise ValueError(
                f"{metric.value} is undefined for {self.method.value} on the {partition.value} partition"
            )
        return MetricSummary.from_values(values)

    def _collect_values(
        self,
        metric: ClassificationMetric,
        partition: EvaluationPartition,
    ) -> list[float]:
        values: list[float | None] = [
            fold.get_metrics(partition).get(metric) for fold in self.folds
        ]
        return [value for value in values if value is not None]
