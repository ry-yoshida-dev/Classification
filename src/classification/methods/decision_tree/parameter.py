from dataclasses import dataclass
from typing import override

from sklearn.tree import DecisionTreeClassifier

from ...class_weighting import ClassWeighting
from ...defaults import DEFAULT_RANDOM_STATE
from ...feature_scaling import FeatureScaling
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...parameter_validator import ParameterValidator
from ...processors import ProbabilisticClassificationProcessor
from ...tree_criterion import TreeCriterion


@dataclass(frozen=True)
class DecisionTreeParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's DecisionTreeClassifier.

    Attributes
    ----------
    criterion : TreeCriterion
        Split quality criterion.
    max_depth : int | None
        Maximum tree depth; no limit when None.
    min_samples_split : int
        Minimum number of samples required to split a node.
    min_samples_leaf : int
        Minimum number of samples required at a leaf.
    max_leaf_nodes : int | None
        Maximum number of leaves; no limit when None.
    ccp_alpha : float
        Complexity parameter for minimal cost-complexity pruning.
    class_weighting : ClassWeighting
        How samples are weighted according to their class.
    random_state : int | None
        Seed for tie-breaking between equally good splits.
    """

    criterion: TreeCriterion = TreeCriterion.GINI
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_leaf_nodes: int | None = None
    ccp_alpha: float = 0.0
    class_weighting: ClassWeighting = ClassWeighting.UNIFORM
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
        )
        ParameterValidator.require_not_nan(
            ccp_alpha=self.ccp_alpha,
        )
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be greater than 0 when set")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than 0")
        if self.max_leaf_nodes is not None and self.max_leaf_nodes < 2:
            raise ValueError("max_leaf_nodes must be at least 2 when set")
        if self.ccp_alpha < 0:
            raise ValueError("ccp_alpha must be non-negative")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted decision tree processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: DecisionTreeClassifier = DecisionTreeClassifier(
            criterion=self.criterion.value,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            ccp_alpha=self.ccp_alpha,
            class_weight=self.class_weighting.to_estimator_argument(),
            random_state=self.random_state,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.DECISION_TREE,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
