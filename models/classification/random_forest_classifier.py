"""
random_forest_classifier.py

Defines the RandomForestClassifierModel class, which implements a
Random Forest algorithm for classification problems.

The model inherits the common interface from the Model base class and
provides methods for training, prediction, and feature analysis.

Feature analysis is based on the feature importance values calculated
by the Random Forest. These values indicate the relative contribution
of each feature to the model's decision-making process.

Inherits from:
    Model

Problem type:
    Classification

Used by:
    ModelFactory
    MachineLearningPipeline
"""




from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from models.model import Model
from models.problem_type import ProblemType




class RandomForestClassifierModel(Model):

    problem_type = ProblemType.CLASSIFICATION

    # Use 100 decision trees to providew a stable ensemble.
    # A fixed random state (seed) makes experiments reproducible.
    def __init__(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )


    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(
            X_train,
            y_train
        )


    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(
            X_test
        )
    

    def get_feature_analysis(self) -> np.ndarray:

        # Random Forest calculates feature importance by looking at
        # how much each feature contributes to decisions across
        #  the trees in the forest.
        return self.model.feature_importances_

