from __future__ import annotations

from enum import Enum
from typing import Literal


class ClassWeighting(Enum):
    """
    How training samples are weighted according to their class.

    Attributes
    ----------
    UNIFORM
        Every sample has the same weight.
    BALANCED
        Weights are inversely proportional to class frequencies.
    """

    UNIFORM = "uniform"
    BALANCED = "balanced"

    def to_estimator_argument(self) -> Literal["balanced"] | None:
        """
        Convert to the scikit-learn ``class_weight`` argument.

        Returns
        -------
        Literal["balanced"] | None
            The value to pass as ``class_weight``.
        """
        match self:
            case ClassWeighting.UNIFORM:
                return None
            case ClassWeighting.BALANCED:
                return "balanced"
