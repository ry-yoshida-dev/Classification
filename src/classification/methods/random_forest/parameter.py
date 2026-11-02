from dataclasses import dataclass
from typing import override

from sklearn.ensemble import RandomForestClassifier

from ...class_weighting import ClassWeighting
from ...defaults import DEFAULT_RANDOM_STATE
from ...feature_scaling import FeatureScaling
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...parameter_validator import ParameterValidator
from ...processors import ProbabilisticClassificationProcessor
from ...tree_criterion import TreeCriterion
from .max_features import RandomForestMaxFeatures


@dataclass(frozen=True)
class RandomForestParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's RandomForestClassifier.

    Attributes
    ----------
    n_estimators : int
        Number of trees.
    criterion : TreeCriterion
        Split quality criterion.
    max_depth : int | None
        Maximum tree depth; no limit when None.
    min_samples_split : int
        Minimum number of samples required to split a node.
    min_samples_leaf : int
        Minimum number of samples required at a leaf.
    max_features : RandomForestMaxFeatures | None
        Features considered per split; all features when None.
    is_bootstrapped : bool
        Whether each tree is trained on a bootstrap sample.
    class_weighting : ClassWeighting
        How samples are weighted according to their class.
    n_jobs : int | None
        Number of parallel jobs.
    random_state : int | None
        Seed for bootstrapping and feature sampling.
    """

    n_estimators: int = 100
    criterion: TreeCriterion = TreeCriterion.GINI
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: RandomForestMaxFeatures | None = RandomForestMaxFeatures.SQRT
    is_bootstrapped: bool = True
    class_weighting: ClassWeighting = ClassWeighting.UNIFORM
    n_jobs: int | None = None
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be greater than 0 when set")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than 0")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted random forest processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: RandomForestClassifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion.value,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=None if self.max_features is None else self.max_features.value,
            bootstrap=self.is_bootstrapped,
            class_weight=self.class_weighting.to_estimator_argument(),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.RANDOM_FOREST,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
