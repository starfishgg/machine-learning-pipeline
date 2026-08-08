"""
regression_evaluator.py

Contains the evaluator used for regression machine learning
problems.

RegressionEvaluator calculates metrics that measure how close
continuous predictions are to the actual values.

Currently calculates:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R² Score

Input:
    PredictionSet containing predicted and actual values.

Output:
    RegressionMetrics containing evaluation results.

Implements:
    Evaluator
"""




from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

from evaluation.evaluator import Evaluator
from evaluation.regression_metrics import RegressionMetrics
from results.prediction_set import PredictionSet




class RegressionEvaluator(Evaluator):


    def evaluate(self, prediction_set: PredictionSet) -> RegressionMetrics:

        predictions = prediction_set.get_predictions()

        actuals = prediction_set.get_actuals()


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

        return RegressionMetrics(
            mae,
            rmse,
            r2
        )