from enum import Enum


class ClassScoreKind(Enum):
    """
    Meaning of the per-class scores produced by a classifier.

    Attributes
    ----------
    PROBABILITY
        Class membership probabilities; each row sums to one.
    MARGIN
        Unbounded confidence values, comparable only within a class column.
    """

    PROBABILITY = "probability"
    MARGIN = "margin"
