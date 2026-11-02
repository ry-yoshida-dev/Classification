from enum import Enum


class BaselineStrategy(Enum):
    """
    Prediction rule of a feature-independent reference classifier.

    Attributes
    ----------
    PRIOR
        Predicts the most frequent class, with class priors as probabilities.
    MOST_FREQUENT
        Predicts the most frequent class, with probability one.
    STRATIFIED
        Samples predictions from the class priors.
    UNIFORM
        Samples predictions uniformly from the classes.
    """

    PRIOR = "prior"
    MOST_FREQUENT = "most_frequent"
    STRATIFIED = "stratified"
    UNIFORM = "uniform"
