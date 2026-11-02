from dataclasses import dataclass

from ..types import ClassLabel


@dataclass(frozen=True)
class PerClassMetrics:
    """
    Performance metrics of a single class.

    Attributes
    ----------
    label : ClassLabel
        The class label.
    precision : float
        Precision; 0 when the class is never predicted.
    recall : float
        Recall; 0 when the class has no true samples.
    f1 : float
        F1 score.
    support : int
        Number of true samples of the class.
    roc_auc : float | None
        One-vs-rest area under the ROC curve; None when the class has no
        positive or no negative samples.
    average_precision : float | None
        One-vs-rest average precision; None when the class has no true samples.
    """

    label: ClassLabel
    precision: float
    recall: float
    f1: float
    support: int
    roc_auc: float | None
    average_precision: float | None
