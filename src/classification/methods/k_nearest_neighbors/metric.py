from enum import Enum


class KNearestNeighborsMetric(Enum):
    """
    Distance metric for scikit-learn's k-nearest neighbors estimators.

    Attributes
    ----------
    EUCLIDEAN
        Euclidean distance.
    MANHATTAN
        Manhattan distance.
    CHEBYSHEV
        Chebyshev distance.
    COSINE
        Cosine distance; requires the AUTO or BRUTE algorithm.
    """

    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    COSINE = "cosine"
