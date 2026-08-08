"""
classification_evaluator.py

Contains the evaluator used for classification machine learning
problems.

ClassificationEvaluator calculates metrics that measure how
well a model predicts discrete categories.

Currently calculates:
    - Accuracy
    - Confusion Matrix

Input:
    PredictionSet containing predicted and actual class labels.

Output:
    ClassificationMetrics containing evaluation results.

Implements:
    Evaluator
"""




from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

from evaluation.evaluator import Evaluator
from evaluation.classification_metrics import ClassificationMetrics
from results.prediction_set import PredictionSet




class ClassificationEvaluator(Evaluator):

    def evaluate(
            self,
            prediction_set: PredictionSet
    ) -> ClassificationMetrics:

        accuracy = accuracy_score(
            prediction_set.get_actuals(),
            prediction_set.get_predictions()
        )

        confusion_matrix_values = confusion_matrix(
            prediction_set.get_actuals(),
            prediction_set.get_predictions()
        )

        return ClassificationMetrics(
            accuracy,
            confusion_matrix_values
        )
