from enum import Enum


class TreeCriterion(Enum):
    """
    Split quality criterion, shared by decision trees and random forests.

    Attributes
    ----------
    GINI
        Gini impurity.
    ENTROPY
        Shannon information gain.
    LOG_LOSS
        Log loss, equivalent to Shannon information gain.
    """

    GINI = "gini"
    ENTROPY = "entropy"
    LOG_LOSS = "log_loss"
