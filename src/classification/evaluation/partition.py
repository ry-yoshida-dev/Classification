from enum import Enum


class EvaluationPartition(Enum):
    """
    Which samples of a fold a metric was measured on.

    Attributes
    ----------
    TRAIN
        The samples the model was fitted on; optimistic by nature.
    TEST
        The held-out samples; the estimate of generalization performance.
    """

    TRAIN = "train"
    TEST = "test"
