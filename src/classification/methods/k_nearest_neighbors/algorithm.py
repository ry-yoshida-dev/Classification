from enum import Enum


class KNearestNeighborsAlgorithm(Enum):
    """
    Neighbor search backend for scikit-learn's k-nearest neighbors estimators.

    Attributes
    ----------
    AUTO
        Choose a backend from the data.
    BALL_TREE
        Ball tree.
    KD_TREE
        KD tree.
    BRUTE
        Brute-force search.
    """

    AUTO = "auto"
    BALL_TREE = "ball_tree"
    KD_TREE = "kd_tree"
    BRUTE = "brute"
