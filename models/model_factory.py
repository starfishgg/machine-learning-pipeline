from models.model_type import ModelType

from models.classification.random_forest_classifier import (
    RandomForestClassifierModel
)

from models.classification.logistic_regression_classifier import (
    LogisticRegressionClassifierModel
)

from models.regression.linear_regression import LinearRegressionModel
from models.regression.random_forest_regressor import RandomForestRegressorModel




class ModelFactory:

    @staticmethod
    def create(model_type):

        # Keep model creation in one place.
        # This means the rest of the application does not need to know
        # which Python class represents each model.

        if model_type == ModelType.RANDOM_FOREST_CLASSIFIER:
            return RandomForestClassifierModel()
        
        elif model_type == ModelType.LOGISTIC_REGRESSION_CLASSIFIER:
            return LogisticRegressionClassifierModel()
        
        elif model_type == ModelType.LINEAR_REGRESSION:
            return LinearRegressionModel()
        
        elif model_type == ModelType.RANDOM_FOREST_REGRESSOR:
            return RandomForestRegressorModel()
        
        else:
            raise ValueError(
                f"Unknown model: {model_type} must be added to model_factory"
            )
        