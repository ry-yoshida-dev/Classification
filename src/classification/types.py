"""Shared type aliases for classification data."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type NumericArray = NDArray[np.integer | np.floating]
type FloatArray = NDArray[np.floating]
type LabelArray = NDArray[np.integer | np.bool_ | np.str_]
type IndexArray = NDArray[np.intp]
type CountArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
type GroupArray = NDArray[np.integer | np.str_]
type ClassLabel = int | str
