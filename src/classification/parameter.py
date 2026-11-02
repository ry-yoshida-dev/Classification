from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .processor import ClassificationProcessor


class ClassificationParameters(ABC):
    """
    Base class for classification parameters that build a processor.
    """

    @abstractmethod
    def build_processor(self) -> ClassificationProcessor:
        """
        Build an unfitted classification processor.

        Returns
        -------
        ClassificationProcessor
            The classification processor.
        """
