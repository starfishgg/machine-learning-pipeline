

class Prediction:

    def __init__(self, predicted, actual):

        # Store normal Python numbers instead of numpy values.
        # Makes printing and saving results easier later.
        self.predicted = float(predicted)
        self.actual = float(actual)


    def error(self):
        return self.predicted - self.actual


    # This is used for regression (scalar predictions)
    def absolute_error(self):
        return abs(
            self.error()
        )
    

    # This is used for classification (yes/no predictions)
    def is_correct(self):
        return self.predicted == self.actual
    

    def show(self):
        print(
            f"Predicted: {self.predicted:.3f}, Actual: {self.actual:.3f}"
        )
