from dataclasses import dataclass
from typing import override

from sklearn.naive_bayes import GaussianNB

from ...feature_scaling import FeatureScaling
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...parameter_validator import ParameterValidator
from ...processors import ProbabilisticClassificationProcessor


@dataclass(frozen=True)
class GaussianNaiveBayesParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's GaussianNB.

    Attributes
    ----------
    var_smoothing : float
        Fraction of the largest feature variance added to all variances.
    """

    var_smoothing: float = 1e-9

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_not_nan(
            var_smoothing=self.var_smoothing,
        )
        if self.var_smoothing < 0:
            raise ValueError("var_smoothing must be non-negative")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted Gaussian naive Bayes processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.GAUSSIAN_NAIVE_BAYES,
            estimator=FeatureScaling.NONE.build_pipeline(
                GaussianNB(var_smoothing=self.var_smoothing)
            ),
        )
