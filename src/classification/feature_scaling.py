from __future__ import annotations

from enum import Enum

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class FeatureScaling(Enum):
    """
    Feature scaling fitted on training data only, as part of the estimator.

    Scaling inside the pipeline means that test folds never influence the
    scaling statistics during evaluation.

    Attributes
    ----------
    NONE
        Features are used as they are.
    STANDARD
        Zero mean and unit variance.
    MIN_MAX
        Rescaled to the [0, 1] range.
    ROBUST
        Centered on the median and scaled by the interquartile range.
    """

    NONE = "none"
    STANDARD = "standard"
    MIN_MAX = "min_max"
    ROBUST = "robust"

    def build_pipeline(self, estimator: object) -> Pipeline:
        """
        Build a pipeline that applies this scaling before an estimator.

        Parameters
        ----------
        estimator : object
            The scikit-learn estimator to run on the scaled features.

        Returns
        -------
        Pipeline
            A pipeline with a ``scaler`` step and an ``estimator`` step.
        """
        return Pipeline([("scaler", self._build_scaler()), ("estimator", estimator)])

    def _build_scaler(self) -> object:
        match self:
            case FeatureScaling.NONE:
                return "passthrough"
            case FeatureScaling.STANDARD:
                return StandardScaler()
            case FeatureScaling.MIN_MAX:
                return MinMaxScaler()
            case FeatureScaling.ROBUST:
                return RobustScaler()
