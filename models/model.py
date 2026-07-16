from abc import ABC, abstractmethod

from models.problem_type import ProblemType



class Model(ABC):

    problem_type = None

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def get_feature_analysis(self):

        # Not every model explains its decisions in the same way.
        # Individual models can override this if they support it.
        raise NotImplementedError(
            "This model does not support feature importance"
        )