"""
random_forest_regressor.py

Defines the RandomForestRegressorModel class, which implements a
Random Forest algorithm for regression problems.

The model inherits the common interface from the Model base class and
provides methods for training, prediction, and feature analysis.

Feature analysis is based on the feature importance values calculated
by the Random Forest. These values indicate the relative contribution
of each feature to the model's decision-making process.

Inherits from:
    Model

Problem type:
    Regression

Used by:
    ModelFactory
    MachineLearningPipeline
"""




from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

from models.model import Model
from models.problem_type import ProblemType




class RandomForestRegressorModel(Model):

    problem_type = ProblemType.REGRESSION
    

    def __init__(self) -> None:

        # Use 100 decision trees to provide a stable ensemble.
        # A fixed random state (seed) makes experiments reproducible.
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )


    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:

        # Train using the known target values.
        self.model.fit(
            X_train,
            y_train
        )


    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        
        # Predict continous target values for data
        # the model has not seen during training.
        return self.model.predict(
            X_test
        )
    

    def get_feature_analysis(self) -> np.ndarray:

        # Random forest calculates feature importance based on
        # how much each feature contributes to decisions across
        # the trees in the forest.
        return self.model.feature_importances_