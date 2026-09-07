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
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
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

        precision = precision_score(
            prediction_set.get_actuals(),
            prediction_set.get_predictions(),
            zero_division=0
        )

        recall = recall_score(
            prediction_set.get_actuals(),
            prediction_set.get_predictions(),
            zero_division=0
        )

        f1 = f1_score(
            prediction_set.get_actuals(),
            prediction_set.get_predictions(),
            zero_division=0
        )

        confusion_matrix_values = confusion_matrix(
            prediction_set.get_actuals(),
            prediction_set.get_predictions()
        )

        return ClassificationMetrics(
            accuracy,
            precision,
            recall,
            f1,
            confusion_matrix_values
        )
