from results.prediction import Prediction



class PredictionResult:

    def __init__(self, predictions, actual):
        
        self.predictions = []

        for predicted, actual_value in zip(
            predictions,
            actual
        ):
            self.predictions.append(
                Prediction(
                    predicted,
                    actual_value
                )
            )

    
    def show_predictions(self, count=10):

        for prediction in self.predictions[:count]:
            prediction.show()


    def count(self):
        return len(self.predictions)


    def get_errors(self):
        
       return [
           prediction
           for prediction in self.predictions
           if not prediction.is_correct()
       ]
    
    
    def get_correct_predictions(self):

       return [
           prediction
           for prediction in self.predictions
           if prediction.is_correct()
       ]
    

    def get_predictions(self):

        return [
            float(prediction.predicted)
            for prediction in self.predictions
        ]
    

    def get_actuals(self):

        return [
            prediction.actual
            for prediction in self.predictions
        ]