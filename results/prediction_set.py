"""
prediction_set.py

Stores the predictions produced by a machine learning model.

A PredictionSet pairs each predicted value with its corresponding
actual target value by creating individual Prediction objects.

This provides a common representation of predictions for both
classification and regression models.

Created by:
    MachineLearningPipeline

Used by:
    ModelResult
    ClassificationEvaluator
    RegressionEvaluator
"""




from collections.abc import Sequence

from results.prediction import Prediction




class PredictionSet:

    def __init__(
            self,
            predictions: Sequence[float],
            actual_values: Sequence[float]
    ) -> None:

        # Using Sequence rather than list here because these values can come from different libraries. Model predictions are usually returned as NumPy arrays, while actual values may be pandas Series.
        # Sequence allows both to be accepted without requiring conversion before creating the Prediction objects.

        # Convert each predition/acttual pair into a Prediction object.
        # This gives the rest of the application a consistent format
        # regardless of whether the values come from NumPy or pandas.
        self.predictions: list[Prediction] = []

        for predicted_value, actual_value in zip(
            predictions,
            actual_values
        ):
            self.predictions.append(
                Prediction(
                    predicted_value,
                    actual_value
                )
            )

    
    def show_predictions(self, count: int=10) -> None:

        for prediction in self.predictions[:count]:
            prediction.show()


    def count(self) -> int:
        return len(self.predictions)


    # Return predictions that were classified incorrectly.
    def get_errors(self) -> list[Prediction]:
        
       return [
           prediction
           for prediction in self.predictions
           if not prediction.is_correct()
       ]
    

    # Return predictions that were classified correctly.
    def get_correct_predictions(self) -> list[Prediction]:

       return [
           prediction
           for prediction in self.predictions
           if prediction.is_correct()
       ]
    

    # Returns the predicted values (as floats).
    def get_predictions(self) -> list[float]:

        return [
            float(prediction.predicted_value)
            for prediction in self.predictions
        ]
    

    # Return the actual target values (as floats)
    def get_actuals(self) -> list[float]:

        return [
            prediction.actual_value
            for prediction in self.predictions
        ]

