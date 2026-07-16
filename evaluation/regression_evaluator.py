from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

from evaluation.evaluator import Evaluator
from results.regression_evaluation_result import RegressionEvaluationResult


class RegressionEvaluator(Evaluator):


    def evaluate(self, prediction_result):

        predictions = prediction_result.get_predictions()

        actuals = prediction_result.get_actuals()


        mae = mean_absolute_error(
            actuals,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                actuals,
                predictions
            )
        )

        r2 = r2_score(
            actuals,
            predictions
        )

        return RegressionEvaluationResult(
            mae,
            rmse,
            r2
        )