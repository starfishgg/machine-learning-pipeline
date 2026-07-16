from models.model_type import ModelType
from datasets.titanic import TitanicDataset
from datasets.california_housing import CaliforniaHousingDataset
from models.model_factory import ModelFactory
# from pipeline.pipeline import MachineLearningPipeline
from pipeline.experiment_runner import ExperimentRunner

from evaluation.classification_evaluator import ClassificationEvaluator




def run_titanic():
    print("=" * 50)
    print("TITANIC SURVIVAL")
    print("=" * 50)
    print()

    # The dataset knows how to Load and preprocess itself.
    # The pipeline doesn't care if this is Titanic, Housing, etc.    
    dataset = TitanicDataset(
        "data/titanic/train.csv"
    )

    dataset.load()
    dataset.preprocess()
    dataset.validate()
    dataset.describe()

    experiment = ExperimentRunner(
        dataset,
        [
            ModelType.RANDOM_FOREST_CLASSIFIER,
            ModelType.LOGISTIC_REGRESSION_CLASSIFIER
        ]
    )

    results = experiment.run()
    
    results.show_comparison()
    results.show_feature_analysis()

    results.compare_predictions(
        ModelType.RANDOM_FOREST_CLASSIFIER,
        ModelType.LOGISTIC_REGRESSION_CLASSIFIER
    )

    return 1, 2




def run_california_housing():
    print("=" * 50)
    print("CALIFORNIA HOUSING")
    print("=" * 50)
    print()

    dataset = CaliforniaHousingDataset()

    dataset.load()
    dataset.preprocess()
    dataset.validate()
    dataset.describe()

    runner = ExperimentRunner(
        dataset,
        [
            ModelType.LINEAR_REGRESSION,
            ModelType.RANDOM_FOREST_REGRESSOR
        ]
    )

    results = runner.run()

    results.show_comparison()
    results.show_feature_analysis()

    results.compare_predictions(
        ModelType.LINEAR_REGRESSION,
        ModelType.RANDOM_FOREST_REGRESSOR
    )

    return 1, 2




def main():
    datasets_processed = 0
    models_evaluated = 0

    d, m = run_california_housing()
    datasets_processed += d
    models_evaluated += m
    
    d, m = run_titanic()
    datasets_processed += d
    models_evaluated += m

    print("=" * 50)
    print("EXPERIMENT COMPLETE")
    print("=" * 50)
    print()

    print(f"Datasets processed : {datasets_processed}")
    print(f"Models evaluated   : {models_evaluated}")





if __name__ == "__main__":
    main()
