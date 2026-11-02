# methods

## Overview

Parameter classes and enums for each classification method. Every parameter class is frozen, validates its fields on construction, and builds an unfitted processor via `build_processor()`.

## Components

| Component | Description |
|-----------|-------------|
| [logistic_regression/](./logistic_regression/README.md) | Logistic regression parameters and solvers. |
| [svc/](./svc/README.md) | Support vector classification parameters, kernels, and gamma rules. |
| [k_nearest_neighbors/](./k_nearest_neighbors/README.md) | K-nearest neighbors parameters and related enums. |
| [decision_tree/](./decision_tree/README.md) | Decision tree parameters. |
| [random_forest/](./random_forest/README.md) | Random forest parameters and feature sampling rules. |
| [hist_gradient_boosting/](./hist_gradient_boosting/README.md) | Histogram gradient boosting parameters. |
| [gaussian_naive_bayes/](./gaussian_naive_bayes/README.md) | Gaussian naive Bayes parameters. |
| [baseline/](./baseline/README.md) | Feature-independent reference classifier. |
