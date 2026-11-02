from __future__ import annotations

from enum import Enum

from .class_score_kind import ClassScoreKind


class ClassificationMetric(Enum):
    """
    Scalar performance metric of a classifier.

    Attributes
    ----------
    ACCURACY
        Fraction of correct predictions.
    BALANCED_ACCURACY
        Mean recall over the classes present in the true labels.
    MACRO_PRECISION
        Unweighted mean precision over classes.
    MACRO_RECALL
        Unweighted mean recall over classes.
    MACRO_F1
        Unweighted mean F1 over classes.
    WEIGHTED_F1
        Support-weighted mean F1 over classes.
    MATTHEWS_CORRELATION
        Matthews correlation coefficient; undefined when only one class occurs.
    COHEN_KAPPA
        Cohen's kappa; undefined when only one class occurs.
    ROC_AUC
        Macro one-vs-rest area under the ROC curve; undefined without scores
        for at least one class that has both positive and negative samples.
    AVERAGE_PRECISION
        Macro one-vs-rest average precision.
    LOG_LOSS
        Cross-entropy loss; defined for probabilistic classifiers only.
    """

    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    MACRO_PRECISION = "macro_precision"
    MACRO_RECALL = "macro_recall"
    MACRO_F1 = "macro_f1"
    WEIGHTED_F1 = "weighted_f1"
    MATTHEWS_CORRELATION = "matthews_correlation"
    COHEN_KAPPA = "cohen_kappa"
    ROC_AUC = "roc_auc"
    AVERAGE_PRECISION = "average_precision"
    LOG_LOSS = "log_loss"

    @property
    def is_higher_better(self) -> bool:
        """
        Check the direction in which this metric improves.

        Returns
        -------
        bool
            True if larger values indicate better performance.
        """
        match self:
            case ClassificationMetric.LOG_LOSS:
                return False
            case _:
                return True

    def is_defined_for(self, kind: ClassScoreKind) -> bool:
        """
        Check whether this metric can be computed from scores of a kind.

        Parameters
        ----------
        kind : ClassScoreKind
            The kind of the per-class scores.

        Returns
        -------
        bool
            False if no evaluation with these scores can define this metric.
        """
        match self:
            case ClassificationMetric.LOG_LOSS:
                return kind == ClassScoreKind.PROBABILITY
            case _:
                return True
