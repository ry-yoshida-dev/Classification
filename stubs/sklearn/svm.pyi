from typing import Literal

class SVC:
    C: float
    kernel: str
    gamma: str | float
    max_iter: int

    def __init__(
        self,
        *,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: str | float = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        class_weight: Literal["balanced"] | None = None,
        max_iter: int = -1,
    ) -> None: ...
