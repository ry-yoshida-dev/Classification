class KNeighborsClassifier:
    n_neighbors: int
    weights: str
    metric: str

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        n_jobs: int | None = None,
    ) -> None: ...
