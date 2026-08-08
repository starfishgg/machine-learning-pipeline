"""
pipeline.py

Defines the MachineLearningPipeline class.

The MachineLearningPipeline coordinates the complete machine learning
workflow for a single model. It takes a dataset through loading,
preprocessing, validation, training, and prediction.

Workflow:

    Dataset
        |
        v
    Load data
        |
        v
    Preprocess data
        |
        v
    Validate dataset
        |
        v
    Train model
        |
        v
    Generate predictions
        |
        v
    Return PredictionSet

The pipeline is responsible for executing one model run. Comparing
multiple models is handled separately by ExperimentRunner.

Created by:
    ExperimentRunner

Uses:
    Dataset
    Model
    PredictionSet
"""




from sklearn.model_selection import train_test_split

from results.prediction_set import PredictionSet
from datasets.dataset import Dataset
from models.model import Model




class MachineLearningPipeline:

    def __init__(self, dataset: Dataset, model: Model) -> None:
        self.dataset = dataset
        self.model = model

    
    def run(self) -> PredictionSet:

        # Load the raw dataset.
        self.dataset.load()

        # Clean and transform the data ready for the model.
        self.dataset.preprocess()

        # Make sure the dataset is ready before training
        self.dataset.validate()

        # Separate the processed dataset into model features
        # and the target value we want the model to predict.
        features  = self.dataset.get_features()
        target = self.dataset.get_target()

        # Split the data into training and testing sets.
        # This belongs in the pieline rather than the Dataset because
        # train/test splitting is part of the model exeriment,
        # not dataset-specific preprocessing
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42
        )

        # Train the model using only the training data.
        self.model.train(
            X_train,
            y_train
        )

        # Generate predictions using the unseen test data.
        test_predictions = self.model.predict(
            X_test
        )

        # Store the predictions alongside the actual test values
        # so they can be evaluated later.
        return PredictionSet(
            test_predictions,
            y_test
        )

