from dataclasses import dataclass
from typing import override

from sklearn.dummy import DummyClassifier

from ...defaults import DEFAULT_RANDOM_STATE
from ...feature_scaling import FeatureScaling
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...parameter_validator import ParameterValidator
from ...processors import ProbabilisticClassificationProcessor
from .strategy import BaselineStrategy


@dataclass(frozen=True)
class BaselineParameters(ClassificationParameters):
    """
    Parameters for a reference classifier that ignores the features.

    A model is only useful if it beats this reference, which is what makes
    numbers like accuracy interpretable on imbalanced data.

    Attributes
    ----------
    strategy : BaselineStrategy
        The prediction rule.
    random_state : int | None
        Seed for the sampling strategies.
    """

    strategy: BaselineStrategy = BaselineStrategy.PRIOR
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(random_state=self.random_state)

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted baseline processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: DummyClassifier = DummyClassifier(
            strategy=self.strategy.value,
            random_state=self.random_state,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.BASELINE,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
