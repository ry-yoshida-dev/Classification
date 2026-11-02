from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..types import BoolArray, ClassLabel, CountArray, FloatArray, LabelArray
from .class_score_kind import ClassScoreKind
from .class_scores import ClassScores
from .confusion_matrix import ConfusionMatrix
from .metric import ClassificationMetric
from .per_class import PerClassMetrics


@dataclass(frozen=True)
class MetricValues:
    """
    Performance of a classifier on one set of labeled samples.

    Macro averages run over the classes that occur in the true or predicted
    labels, so a class absent from both does not distort them.

    Attributes
    ----------
    num_samples : int
        Number of evaluated samples.
    accuracy : float
        Fraction of correct predictions.
    balanced_accuracy : float
        Mean recall over the classes present in the true labels.
    macro_precision : float
        Unweighted mean precision.
    macro_recall : float
        Unweighted mean recall.
    macro_f1 : float
        Unweighted mean F1.
    weighted_f1 : float
        Support-weighted mean F1.
    matthews_correlation : float | None
        Matthews correlation coefficient; None when only one class occurs.
    cohen_kappa : float | None
        Cohen's kappa; None when only one class occurs.
    roc_auc : float | None
        Macro one-vs-rest ROC AUC over the classes where it is defined.
    average_precision : float | None
        Macro one-vs-rest average precision over the classes where it is defined.
    log_loss : float | None
        Cross-entropy loss; None unless the scores are probabilities.
    per_class : tuple[PerClassMetrics, ...]
        Metrics of each class known to the classifier.
    confusion_matrix : ConfusionMatrix
        Counts of true against predicted classes.
    """

    num_samples: int
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    matthews_correlation: float | None
    cohen_kappa: float | None
    roc_auc: float | None
    average_precision: float | None
    log_loss: float | None
    per_class: tuple[PerClassMetrics, ...]
    confusion_matrix: ConfusionMatrix

    @classmethod
    def compute(
        cls,
        y_true: LabelArray,
        y_pred: LabelArray,
        scores: ClassScores,
    ) -> MetricValues:
        """
        Compute all metrics from labels and per-class scores.

        Parameters
        ----------
        y_true : LabelArray
            The true labels with shape (n_samples,).
        y_pred : LabelArray
            The predicted labels with shape (n_samples,).
        scores : ClassScores
            The per-class scores with shape (n_samples, n_classes).

        Returns
        -------
        MetricValues
            The computed metrics.

        Raises
        ------
        ValueError
            If the inputs are empty or misaligned, or if a label is not one of
            the scored classes.
        """
        cls._validate_inputs(y_true, y_pred, scores)
        classes: LabelArray = scores.classes
        matrix: ConfusionMatrix = ConfusionMatrix.from_predictions(
            y_true, y_pred, classes
        )
        per_class: tuple[PerClassMetrics, ...] = cls._compute_per_class(
            y_true, y_pred, scores
        )
        support: CountArray = matrix.counts.sum(axis=1)
        predicted: CountArray = matrix.counts.sum(axis=0)
        is_class_active: BoolArray = (support > 0) | (predicted > 0)
        f1: FloatArray = np.array([item.f1 for item in per_class], dtype=np.float64)
        return cls(
            num_samples=y_true.size,
            accuracy=float(accuracy_score(y_true, y_pred)),
            balanced_accuracy=cls._compute_balanced_accuracy(matrix),
            macro_precision=cls._mean_over_active(
                [item.precision for item in per_class], is_class_active
            ),
            macro_recall=cls._mean_over_active(
                [item.recall for item in per_class], is_class_active
            ),
            macro_f1=cls._mean_over_active(
                [item.f1 for item in per_class], is_class_active
            ),
            weighted_f1=float(np.average(f1, weights=support)),
            matthews_correlation=cls._compute_matthews_correlation(y_true, y_pred),
            cohen_kappa=cls._compute_cohen_kappa(y_true, y_pred),
            roc_auc=cls._mean_of_defined([item.roc_auc for item in per_class]),
            average_precision=cls._mean_of_defined(
                [item.average_precision for item in per_class]
            ),
            log_loss=cls._compute_log_loss(y_true, scores),
            per_class=per_class,
            confusion_matrix=matrix,
        )

    def get(self, metric: ClassificationMetric) -> float | None:
        """
        Get the value of a metric.

        Parameters
        ----------
        metric : ClassificationMetric
            The metric to look up.

        Returns
        -------
        float | None
            The value, or None when the metric is undefined for this evaluation.
        """
        match metric:
            case ClassificationMetric.ACCURACY:
                return self.accuracy
            case ClassificationMetric.BALANCED_ACCURACY:
                return self.balanced_accuracy
            case ClassificationMetric.MACRO_PRECISION:
                return self.macro_precision
            case ClassificationMetric.MACRO_RECALL:
                return self.macro_recall
            case ClassificationMetric.MACRO_F1:
                return self.macro_f1
            case ClassificationMetric.WEIGHTED_F1:
                return self.weighted_f1
            case ClassificationMetric.MATTHEWS_CORRELATION:
                return self.matthews_correlation
            case ClassificationMetric.COHEN_KAPPA:
                return self.cohen_kappa
            case ClassificationMetric.ROC_AUC:
                return self.roc_auc
            case ClassificationMetric.AVERAGE_PRECISION:
                return self.average_precision
            case ClassificationMetric.LOG_LOSS:
                return self.log_loss

    @staticmethod
    def _validate_inputs(
        y_true: LabelArray,
        y_pred: LabelArray,
        scores: ClassScores,
    ) -> None:
        if y_true.ndim != 1 or y_true.size == 0:
            raise ValueError(f"y_true must be non-empty and 1-D, got {y_true.shape}")
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"y_pred must have shape {y_true.shape}, got {y_pred.shape}"
            )
        if scores.values.shape[0] != y_true.size:
            raise ValueError(
                f"scores must have {y_true.size} rows, got {scores.values.shape[0]}"
            )
        for name, labels in (("y_true", y_true), ("y_pred", y_pred)):
            unknown: LabelArray = np.setdiff1d(labels, scores.classes)
            if unknown.size > 0:
                raise ValueError(
                    f"{name} contains labels the classifier was not fitted on: {unknown.tolist()}"
                )

    @staticmethod
    def _compute_per_class(
        y_true: LabelArray,
        y_pred: LabelArray,
        scores: ClassScores,
    ) -> tuple[PerClassMetrics, ...]:
        precision: FloatArray
        recall: FloatArray
        f1: FloatArray
        support: CountArray
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=scores.classes,
            average=None,
            zero_division=0,
        )
        labels: list[ClassLabel] = cast(list[ClassLabel], scores.classes.tolist())
        precisions: list[float] = cast(list[float], precision.tolist())
        recalls: list[float] = cast(list[float], recall.tolist())
        f1_scores: list[float] = cast(list[float], f1.tolist())
        supports: list[int] = cast(list[int], support.tolist())
        return tuple(
            PerClassMetrics(
                label=label,
                precision=precisions[position],
                recall=recalls[position],
                f1=f1_scores[position],
                support=supports[position],
                roc_auc=MetricValues._compute_roc_auc(
                    np.equal(y_true, label), scores.values[:, position]
                ),
                average_precision=MetricValues._compute_average_precision(
                    np.equal(y_true, label), scores.values[:, position]
                ),
            )
            for position, label in enumerate(labels)
        )

    @staticmethod
    def _compute_roc_auc(is_positive: BoolArray, scores: FloatArray) -> float | None:
        num_positive: int = int(np.count_nonzero(is_positive))
        if num_positive == 0 or num_positive == is_positive.size:
            return None
        return float(roc_auc_score(is_positive, scores))

    @staticmethod
    def _compute_average_precision(
        is_positive: BoolArray, scores: FloatArray
    ) -> float | None:
        if not np.any(is_positive):
            return None
        return float(average_precision_score(is_positive, scores))

    @staticmethod
    def _compute_balanced_accuracy(matrix: ConfusionMatrix) -> float:
        support: CountArray = matrix.counts.sum(axis=1)
        is_present: BoolArray = support > 0
        recalls: FloatArray = (
            np.diagonal(matrix.counts)[is_present] / support[is_present]
        )
        return float(np.mean(recalls))

    @staticmethod
    def _compute_matthews_correlation(
        y_true: LabelArray, y_pred: LabelArray
    ) -> float | None:
        if np.union1d(y_true, y_pred).size < 2:
            return None
        return float(matthews_corrcoef(y_true, y_pred))

    @staticmethod
    def _compute_cohen_kappa(y_true: LabelArray, y_pred: LabelArray) -> float | None:
        if np.union1d(y_true, y_pred).size < 2:
            return None
        return float(cohen_kappa_score(y_true, y_pred))

    @staticmethod
    def _compute_log_loss(y_true: LabelArray, scores: ClassScores) -> float | None:
        match scores.kind:
            case ClassScoreKind.PROBABILITY:
                return float(log_loss(y_true, scores.values, labels=scores.classes))
            case ClassScoreKind.MARGIN:
                return None

    @staticmethod
    def _mean_over_active(values: list[float], is_class_active: BoolArray) -> float:
        return float(np.mean(np.array(values, dtype=np.float64)[is_class_active]))

    @staticmethod
    def _mean_of_defined(values: list[float | None]) -> float | None:
        defined: list[float] = [value for value in values if value is not None]
        return float(np.mean(defined)) if defined else None
