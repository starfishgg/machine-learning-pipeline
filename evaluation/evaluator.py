"""
evaluator.py

Defines the abstract base class for all model evaluators.

An Evaluator is responsible for calculating performance metrics
from a PredictionSet. Different machine learning problem types
(classification or regression) implement their own evaluation logic.

Created by:
    EvaluatorFactory

Implemented by:
    ClassificationEvaluator
    RegressionEvaluator
"""




from abc import ABC, abstractmethod

from results.prediction_set import PredictionSet
from evaluation.classification_metrics import ClassificationMetrics
from evaluation.regression_metrics import RegressionMetrics




class Evaluator(ABC):

    # Probably not worth making a Metrics base class just to make a cleaner return type here? So we are making a union of type returns.
    @abstractmethod
    def evaluate(
        self,
        prediction_set: PredictionSet
     ) -> ClassificationMetrics | RegressionMetrics:
        pass
