"""
prediction.py

Represents a single prediction made by a machine learning model.

Stores both the predicted value and the corresponding actual value
so that evaluators can perform error analysis.

Prediction values are kept in their original Python-compatible form
rather than being forced into floats. This allows the class to support
both numerical regression predictions and categorical classification
predictions.

Used by:
    PredictionSet
"""




class Prediction:

    def __init__(self, predicted_value, actual_value) -> None:

        # Store normal Python numbers instead of numpy values.
        # Makes printing and saving results easier later.
        # TODO: We probably need to store these as something
        # other than floats if we do any comparisons on, for example:
        # text values, e.g. actual="cat", predicted="dog".
        # for now, I have removed the float() casting.
        self.predicted_value = predicted_value
        self.actual_value = actual_value


    def error(self) -> float:
        return self.predicted_value - self.actual_value


    # This is used for regression (scalar predictions)
    def absolute_error(self) -> float:
        return abs(
            self.error()
        )
    

    # This is used for classification (yes/no predictions)
    def is_correct(self) -> bool:
        return self.predicted_value == self.actual_value
    
    """
    def show(self):
        print(
            f"Predicted: {self.predicted_value:.3f}, Actual: {self.actual_value:.3f}"
        )
    """
