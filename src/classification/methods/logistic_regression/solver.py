from enum import Enum


class LogisticRegressionSolver(Enum):
    """
    Optimization algorithm for scikit-learn's LogisticRegression.

    Attributes
    ----------
    LBFGS
        Limited-memory BFGS; L2 penalty only.
    LIBLINEAR
        Coordinate descent; binary problems only, with L1 or L2 penalty.
    NEWTON_CG
        Newton conjugate gradient; L2 penalty only.
    NEWTON_CHOLESKY
        Newton with Cholesky decomposition; L2 penalty only.
    SAG
        Stochastic average gradient; L2 penalty only.
    SAGA
        SAG variant supporting L1, L2, and elastic-net penalties.
    """

    LBFGS = "lbfgs"
    LIBLINEAR = "liblinear"
    NEWTON_CG = "newton-cg"
    NEWTON_CHOLESKY = "newton-cholesky"
    SAG = "sag"
    SAGA = "saga"
