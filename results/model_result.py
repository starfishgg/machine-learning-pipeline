



class ModelResult:

    def __init__(
            self,
            model,
            prediction_result
    ):
        
        # Store the trainer model because
        # some analysis comes from the model itself
        self.model = model

        # Store the predictions because
        # evaluation comes from these.
        self.prediction_result = prediction_result


    def get_prediction_result(self):
        return self.prediction_result


    def get_predictions(self):
        return self.prediction_result.get_predictions()
    

    def get_actuals(self):
        return self.predition_result.get_actuals()
    

    def get_feature_analysis(self):
        return self.model.get_feature_analysis()
