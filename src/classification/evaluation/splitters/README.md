# splitters

## Overview

Concrete `DatasetSplitter` strategies. Choose by how samples relate to each other: independent samples suit the stratified splitters, correlated samples need group folds, and temporal data needs forward chaining.

## Components

| Component | Description |
|-----------|-------------|
| [stratified_k_fold.py](./stratified_k_fold.py) | `StratifiedKFoldSplitter`, the default. |
| [stratified_holdout.py](./stratified_holdout.py) | `StratifiedHoldoutSplitter`, a single partition. |
| [stratified_group_k_fold.py](./stratified_group_k_fold.py) | `StratifiedGroupKFoldSplitter`, which keeps each group on one side. |
| [time_series.py](./time_series.py) | `TimeSeriesSplitter`, which always tests on later samples. |
