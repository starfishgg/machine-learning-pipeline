from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

from evaluation.evaluator import Evaluator
from results.evaluation_result import EvaluationResult



class ClassificationEvaluator(Evaluator):

    def evaluate(self, result):

        accuracy = accuracy_score(
            result.get_actuals(),
            result.get_predictions()
        )

        matrix = confusion_matrix(
            result.get_actuals(),
            result.get_predictions()
        )

        return EvaluationResult(
            accuracy,
            matrix
        )
    