from typing import Literal

class LogisticRegression:
    C: float
    l1_ratio: float
    solver: str
    max_iter: int
    class_weight: Literal["balanced"] | None

    def __init__(
        self,
        *,
        C: float = 1.0,
        l1_ratio: float = 0.0,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        class_weight: Literal["balanced"] | None = None,
        random_state: int | None = None,
        solver: str = "lbfgs",
        max_iter: int = 100,
    ) -> None: ...
