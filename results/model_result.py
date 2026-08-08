"""
model_result.py

Represents the output of running one model during an experiment.

A ModelResult combines:
- The trained model.
- The PredictionSet produced by that model.

This allows later evaluation and analysis to access both the model
itself and the predictions it generated.

Created by:
    ExperimentRunner

Used by:
    ExperimentResult
"""




import numpy as np

from models.model import Model
from results.prediction_set import PredictionSet




class ModelResult:

    def __init__(
            self,
            model: Model,
            prediction_set: PredictionSet
    ) -> None:
        
        # Store the trained model because some analysis, 
        # such as feature importance, comes directly from the model
        self.model = model

        # Store the predictions produced by the model.
        # These are used later for evaluation and comparison.
        self.prediction_set = prediction_set


    def get_prediction_set(self) -> PredictionSet:
        return self.prediction_set


    def get_predictions(self) -> list[float]:
        return self.prediction_set.get_predictions()
    

    def get_actuals(self) -> list[float]:
        return self.prediction_set.get_actuals()
    

    def get_feature_analysis(self) -> np.ndarray:
        return self.model.get_feature_analysis()
