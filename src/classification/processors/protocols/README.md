# processors/protocols

## Overview

Structural typing protocols for wrapped classifiers. Each file mirrors the corresponding processor module; any object satisfying a protocol can be wrapped, including custom scikit-learn compatible estimators.

## Components

| Component | Description |
|-----------|-------------|
| [probabilistic.py](./probabilistic.py) | `ProbabilisticClassifierLike`, with `fit`, `predict`, and `predict_proba`. |
| [margin.py](./margin.py) | `MarginClassifierLike`, with `fit`, `predict`, and `decision_function`. |
