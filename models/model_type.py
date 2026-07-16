from enum import Enum


class ModelType(Enum):

    # These are the suppoted models in the application.
    # Adding a new model later means adding another option here.

    # Classification models

    RANDOM_FOREST_CLASSIFIER = (
        "random_forest_classifier"
    )

    LOGISTIC_REGRESSION_CLASSIFIER = (
        "logistic_regression_classifier"
    )


    # Regression models

    LINEAR_REGRESSION = (
        "linear_regression"
    )

    RANDOM_FOREST_REGRESSOR = (
        "random_forest_regressor"
    )