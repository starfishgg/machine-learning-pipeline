"""
linear_regression.py

Defines the LinearRegressionModel class, which implements linear
regression for regression problems.

The model inherits the common interface from the Model base class and
provides methods for training, prediction, and feature analysis.

Feature analysis is based on the coefficients learned by the linear
regression model. Each coefficient represents the expected change in
the predicted target associated with a one-unit change in the
corresponding feature, assuming the other features remain constant.

Inherits from:
    Model

Problem type:
    Regression

Used by:
    ModelFactory
    MachineLearningPipeline
"""




from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

from models.model import Model
from models.problem_type import ProblemType




class LinearRegressionModel(Model):

    problem_type = ProblemType.REGRESSION
    

    def __init__(self) -> None:
        self.model = LinearRegression()

    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        
        # Teach the model the relationship between
        # the input features and the target value.
        self.model.fit(
            X_train,
            y_train
        )


    def predict(self, X_test: pd.DataFrame) -> np.ndarray:

        # Return predicted target values for data
        # the model has not seen during training.
        return self.model.predict(
            X_test
        )
    

    def get_feature_analysis(self) -> np.ndarray:

        # Linear regression coefficients show the expected
        # change in the target for a one-unit change in each feature.
        return self.model.coef_
    