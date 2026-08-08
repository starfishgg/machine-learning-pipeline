"""
experiment_runner.py

Coordinates a complete machine learning experiment.

For each requested model:
    1.) Creates a fresh model.
    2.) Creates a separate copy of the dataset.
    3.) Runs the complete machine learning pipeline.
    4.) Stores the model and its prediction set.

Finally returns an ExperimentResult containing
every model run.

Used by:
    main
"""




from models.model_factory import ModelFactory
from pipeline.pipeline import MachineLearningPipeline
from results.experiment_result import ExperimentResult
from results.model_result import ModelResult
from datasets.dataset import Dataset
from models.model_type import ModelType
from models.model import Model




class ExperimentRunner:

    def __init__(
            self, 
            dataset: Dataset, 
            model_types: list[ModelType]
    ) -> None:

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


    def run(self) -> ExperimentResult:

        model_results: dict[ModelType, ModelResult] = {}

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

            # Clone the dataset so each model receives its own
            # independent copy of the data.
            pipeline = MachineLearningPipeline(
                self.dataset.clone(),
                model
            )

            # Run the complete ML pipeline once.
            # The result conttains the predictions made by thsi model
            prediction_set = pipeline.run()

            # Store both the trained model and its predictions so 
            # they can be evaluated and analysed later.
            model_results[model_type] = ModelResult(
                model,
                prediction_set
            )

        # Combine all model results into an experiment result.
        return ExperimentResult(
            model_results,
            self.dataset.get_feature_names()
        )
    
