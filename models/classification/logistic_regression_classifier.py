"""
logistic_regression_classifier.py

Defines the LogisticRegressionClassifierModel class, which implements
logistic regression for classification problems.

The model inherits the common interface from the Model base class and
provides methods for training, prediction, and feature analysis.

Feature analysis is based on the coefficients learned by the logistic
regression model. The coefficients indicate the direction and relative
strength of each feature's relationship with the predicted outcome.

Inherits from:
    Model

Problem type:
    Classification

Used by:
    ModelFactory
    MachineLearningPipeline
"""




from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from models.model import Model
from models.problem_type import ProblemType




class LogisticRegressionClassifierModel(Model):

    problem_type = ProblemType.CLASSIFICATION
    

    def __init__(self) -> None:

        # Logistic regression is a linear classification model that
        # estimates the probability of an observation belonging to a class.
        # max_iter is increased because the default sometimes isn't enough
        # for the algoritm to converge.
        self.model = LogisticRegression(
            max_iter=1000
        )


    def train(self, X_train: pd.DataFrame, y_train: pd.Series):

        # Learn the relationship between the input and the outcomes
        self.model.fit(
            X_train,
            y_train
        )


    def predict(self, X_test: pd.DataFrame) -> np.ndarray:

        # Predict the target values for data the model hasn't seen
        # during training.
        return self.model.predict(
            X_test
        )

    
    def get_feature_analysis(self) -> np.ndarray:

        # Logistic regression coefficients show the direction
        # and strength of each feature's relationship with the target.
        return self.model.coef_[0]

    