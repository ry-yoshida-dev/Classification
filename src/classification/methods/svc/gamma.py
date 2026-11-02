from enum import Enum


class SVCGamma(Enum):
    """
    Automatic kernel coefficient rule for scikit-learn's SVC.

    Attributes
    ----------
    SCALE
        1 / (n_features * X.var()).
    AUTO
        1 / n_features.
    """

    SCALE = "scale"
    AUTO = "auto"
