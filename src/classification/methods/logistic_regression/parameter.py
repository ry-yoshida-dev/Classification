from dataclasses import dataclass
from typing import override

from sklearn.linear_model import LogisticRegression

from ...class_weighting import ClassWeighting
from ...defaults import DEFAULT_RANDOM_STATE
from ...feature_scaling import FeatureScaling
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...parameter_validator import ParameterValidator
from ...processors import ProbabilisticClassificationProcessor
from .solver import LogisticRegressionSolver


@dataclass(frozen=True)
class LogisticRegressionParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's LogisticRegression.

    Attributes
    ----------
    C : float
        Inverse regularization strength.
    l1_ratio : float
        Elastic-net mixing: 0 is pure L2 and 1 is pure L1.
    solver : LogisticRegressionSolver
        The optimization algorithm.
    max_iter : int
        Maximum number of solver iterations.
    tol : float
        Stopping tolerance.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    class_weighting : ClassWeighting
        How samples are weighted according to their class.
    scaling : FeatureScaling
        Feature scaling; regularization and solvers depend on feature scale.
    random_state : int | None
        Seed for the solvers that shuffle the data.
    """

    C: float = 1.0
    l1_ratio: float = 0.0
    solver: LogisticRegressionSolver = LogisticRegressionSolver.LBFGS
    max_iter: int = 100
    tol: float = 1e-4
    is_intercept_fitted: bool = True
    class_weighting: ClassWeighting = ClassWeighting.UNIFORM
    scaling: FeatureScaling = FeatureScaling.STANDARD
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        ParameterValidator.require_not_nan(
            C=self.C,
            tol=self.tol,
        )
        if self.C <= 0:
            raise ValueError("C must be greater than 0")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be in the range [0, 1]")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")
        self._validate_solver_penalty()

    def _validate_solver_penalty(self) -> None:
        match self.solver:
            case LogisticRegressionSolver.SAGA:
                return
            case LogisticRegressionSolver.LIBLINEAR:
                if self.l1_ratio not in (0.0, 1.0):
                    raise ValueError("liblinear requires l1_ratio to be 0 or 1")
            case _:
                if self.l1_ratio != 0.0:
                    raise ValueError(
                        f"solver {self.solver.value} requires l1_ratio to be 0"
                    )

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted logistic regression processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: LogisticRegression = LogisticRegression(
            C=self.C,
            l1_ratio=self.l1_ratio,
            solver=self.solver.value,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.is_intercept_fitted,
            class_weight=self.class_weighting.to_estimator_argument(),
            random_state=self.random_state,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.LOGISTIC_REGRESSION,
            estimator=self.scaling.build_pipeline(estimator),
        )
