"""
model_type.py

Defines the ModelType enumeration used to identify the machine
learning models supported by the application.

ModelType provides a consistent identifier for each model and is used
by ModelFactory to create the corresponding concrete model class.

Adding support for a new model requires adding a new ModelType value
and registering the model with ModelFactory.

Used by:
    ModelFactory
    ExperimentRunner
    main
"""




from enum import Enum




class ModelType(Enum):

    # These are the suppoted identifiers for models in the application.
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