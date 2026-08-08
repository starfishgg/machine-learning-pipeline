"""
model.py

Defines the abstract Model base class used by all machine learning
models in the project.

Model provides a common interface for training, making predictions,
and optionally analysing feature importance. Concrete model classes
implement the training and prediction behaviour for specific
algorithms.

The model also declares the type of machine learning problem it
supports, such as classification or regression.

Implemented by:
    LogisticRegressionClassifierModel
    RandomForestClassifierModel
    LinearRegressionModel
    RandomForestRegressorModel

Used by:
    ModelFactory
    MachineLearningPipeline
    ModelResult
"""




from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from models.problem_type import ProblemType




class Model(ABC):

    problem_type: ProblemType | None = None


    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass


    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        pass


    def get_feature_analysis(self) -> np.ndarray:

        # Not every model explains its decisions in the same way.
        # Individual models can override this if they support it.
        raise NotImplementedError(
            "This model does not support feature importance"
        )