from dataclasses import dataclass
from typing import override

from sklearn.svm import SVC

from ...class_weighting import ClassWeighting
from ...feature_scaling import FeatureScaling
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...parameter_validator import ParameterValidator
from ...processors import MarginClassificationProcessor
from .gamma import SVCGamma
from .kernel import SVCKernel


@dataclass(frozen=True)
class SVCParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's SVC.

    The processor exposes margins, not probabilities, because SVC's built-in
    probability estimation is deprecated in scikit-learn.

    Attributes
    ----------
    C : float
        Inverse regularization strength.
    kernel : SVCKernel
        The kernel function.
    degree : int
        Degree of the polynomial kernel.
    gamma : SVCGamma | float
        Kernel coefficient, either a rule or a positive value.
    coef0 : float
        Independent term of the polynomial and sigmoid kernels.
    tol : float
        Stopping tolerance.
    max_iter : int | None
        Maximum number of solver iterations; no limit when None.
    class_weighting : ClassWeighting
        How samples are weighted according to their class.
    scaling : FeatureScaling
        Feature scaling; kernels depend on feature scale.
    """

    C: float = 1.0
    kernel: SVCKernel = SVCKernel.RBF
    degree: int = 3
    gamma: SVCGamma | float = SVCGamma.SCALE
    coef0: float = 0.0
    tol: float = 1e-3
    max_iter: int | None = None
    class_weighting: ClassWeighting = ClassWeighting.UNIFORM
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            degree=self.degree,
            max_iter=self.max_iter,
        )
        ParameterValidator.require_not_nan(
            C=self.C,
            coef0=self.coef0,
            tol=self.tol,
        )
        if self.C <= 0:
            raise ValueError("C must be greater than 0")
        if self.degree < 0:
            raise ValueError("degree must be non-negative")
        if not isinstance(self.gamma, SVCGamma):
            ParameterValidator.require_not_nan(gamma=self.gamma)
            if self.gamma <= 0:
                raise ValueError("gamma must be greater than 0 when numeric")
        if self.tol <= 0:
            raise ValueError("tol must be greater than 0")
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0 when set")

    @override
    def build_processor(self) -> MarginClassificationProcessor:
        """
        Build an unfitted support vector classification processor.

        Returns
        -------
        MarginClassificationProcessor
            The processor.
        """
        estimator: SVC = SVC(
            C=self.C,
            kernel=self.kernel.value,
            degree=self.degree,
            gamma=self.gamma.value if isinstance(self.gamma, SVCGamma) else self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            max_iter=-1 if self.max_iter is None else self.max_iter,
            class_weight=self.class_weighting.to_estimator_argument(),
        )
        return MarginClassificationProcessor(
            method=ClassificationMethod.SVC,
            estimator=self.scaling.build_pipeline(estimator),
        )
