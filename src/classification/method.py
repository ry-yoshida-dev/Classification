from enum import Enum


class ClassificationMethod(Enum):
    """
    Classification method implemented by a processor.

    Attributes
    ----------
    LOGISTIC_REGRESSION
        Logistic regression.
    SVC
        Support vector classifier.
    K_NEAREST_NEIGHBORS
        K-nearest neighbors classifier.
    DECISION_TREE
        Decision tree classifier.
    RANDOM_FOREST
        Random forest classifier.
    HIST_GRADIENT_BOOSTING
        Histogram-based gradient boosting classifier.
    GAUSSIAN_NAIVE_BAYES
        Gaussian naive Bayes classifier.
    BASELINE
        Feature-independent reference classifier.
    """

    LOGISTIC_REGRESSION = "LogisticRegression"
    SVC = "SVC"
    K_NEAREST_NEIGHBORS = "KNearestNeighbors"
    DECISION_TREE = "DecisionTree"
    RANDOM_FOREST = "RandomForest"
    HIST_GRADIENT_BOOSTING = "HistGradientBoosting"
    GAUSSIAN_NAIVE_BAYES = "GaussianNaiveBayes"
    BASELINE = "Baseline"
