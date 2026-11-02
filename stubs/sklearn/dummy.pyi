class DummyClassifier:
    strategy: str

    def __init__(
        self,
        *,
        strategy: str = "prior",
        random_state: int | None = None,
    ) -> None: ...
