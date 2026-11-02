from typing import Literal

class DecisionTreeClassifier:
    criterion: str
    max_depth: int | None

    def __init__(
        self,
        *,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
        max_leaf_nodes: int | None = None,
        class_weight: Literal["balanced"] | None = None,
        ccp_alpha: float = 0.0,
    ) -> None: ...
