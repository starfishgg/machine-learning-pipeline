from sklearn.model_selection import train_test_split

from datasets.dataset import Dataset
from results.prediction_result import PredictionResult

class MachineLearningPipeline:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    
    def run(self):

        # Load the raw dataset.
        self.dataset.load()

        # Clean and transform the data ready for the model.
        self.dataset.preprocess()

        # Make sure the dataset is ready before training
        self.dataset.validate()

        X = self.dataset.get_features()
        y = self.dataset.get_target()

        # train_test_split here as this is not a preprocessor responsibility
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        self.model.train(
            X_train,
            y_train
        )

        predictions = self.model.predict(
            X_test
        )

        return PredictionResult(
            predictions,
            y_test
        )
    