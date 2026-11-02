from enum import Enum


class RandomForestMaxFeatures(Enum):
    """
    Number of features considered per split in a random forest.

    Use None in RandomForestParameters, not this enum, to consider all features.

    Attributes
    ----------
    SQRT
        Square root of the number of features.
    LOG2
        Base-2 logarithm of the number of features.
    """

    SQRT = "sqrt"
    LOG2 = "log2"
