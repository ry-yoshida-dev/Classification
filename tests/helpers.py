from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from classification import ClassScoreKind, ClassScores, DataSplit
from classification.types import FloatArray, LabelArray, NumericArray
from tests.fixed_splitter import FixedSplitter


def separable_blobs(
    *,
    num_classes: int = 3,
    num_samples_per_class: int = 30,
    seed: int = 0,
) -> tuple[NumericArray, NDArray[np.int64]]:
    """Return well-separated 2-D Gaussian blobs and their integer labels."""
    rng: np.random.Generator = np.random.default_rng(seed)
    features: list[NumericArray] = [
        rng.standard_normal((num_samples_per_class, 2)) + 8.0 * label
        for label in range(num_classes)
    ]
    labels: NDArray[np.int64] = np.repeat(
        np.arange(num_classes, dtype=np.int64), num_samples_per_class
    )
    return np.vstack(features), labels


def noise_features(
    *,
    num_samples: int = 200,
    majority_fraction: float = 0.8,
    seed: int = 0,
) -> tuple[NumericArray, NDArray[np.int64]]:
    """Return features independent of imbalanced binary labels."""
    rng: np.random.Generator = np.random.default_rng(seed)
    num_majority: int = int(num_samples * majority_fraction)
    labels: NDArray[np.int64] = np.array(
        [0] * num_majority + [1] * (num_samples - num_majority), dtype=np.int64
    )
    return rng.standard_normal((num_samples, 3)), labels


def probability_scores(classes: LabelArray, values: list[list[float]]) -> ClassScores:
    """Return probability scores built from nested lists."""
    array: FloatArray = np.array(values, dtype=np.float64)
    return ClassScores(classes, array, ClassScoreKind.PROBABILITY)


def single_class_test_folds() -> tuple[NumericArray, NDArray[np.int64], FixedSplitter]:
    """Return data and splits whose first test fold contains a single class."""
    features: NumericArray = np.arange(40.0).reshape(-1, 1)
    labels: NDArray[np.int64] = np.array(
        [0, 1] * 10 + [0] * 10 + [0, 1] * 5, dtype=np.int64
    )
    train_indices: NDArray[np.intp] = np.arange(20, dtype=np.intp)
    splitter: FixedSplitter = FixedSplitter(
        splits=(
            DataSplit(train_indices, np.arange(20, 30, dtype=np.intp)),
            DataSplit(train_indices, np.arange(30, 40, dtype=np.intp)),
        )
    )
    return features, labels, splitter
