from enum import Enum


class KNearestNeighborsWeighting(Enum):
    """
    How neighbors contribute to a prediction.

    Attributes
    ----------
    UNIFORM
        All neighbors count equally.
    DISTANCE
        Neighbors count by the inverse of their distance.
    """

    UNIFORM = "uniform"
    DISTANCE = "distance"
