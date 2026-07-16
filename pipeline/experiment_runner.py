from models.model_factory import ModelFactory
from pipeline.pipeline import MachineLearningPipeline
from results.experiment_result import ExperimentResult
from results.model_result import ModelResult



class ExperimentRunner:

    def __init__(self, dataset, model_types):

        # Store the dataset once.
        # Every model should be tested against the same data,
        # otherwise we aren't making a fair comparison.
        self.dataset = dataset

        # This is the list of models we want to compare.
        # Example:
        # [
        #       ModelType.RANDOM_FOREST,
        #       ModelType.LOGISTIC_REGRESSION
        # ]
        self.model_types = model_types


    def run(self):

        results = {}

        for model_type in self.model_types:

            # Create a brand new model for this experiment.
            # We don't want a trained model from a previous run.
            model = ModelFactory.create(
                model_type
            )


            # A pipeline represents one complete ML run:
            #
            # Dataset
            #    |
            #    v
            # Preprocessing
            #    |
            #    v
            # Training
            #    |
            #    v
            # Predictions
            #
            pipeline = MachineLearningPipeline(
                self.dataset.clone(),
                model
            )

            # Run the complete ML pipeline once.
            # The result conttains the predictions made by thsi model
            prediction_result = pipeline.run()
            
            results[model_type] = ModelResult(
                model,
                prediction_result
            )

        return ExperimentResult(
            results,
            self.dataset.get_feature_names()
        )
    
