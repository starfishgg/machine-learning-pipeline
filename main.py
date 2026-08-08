"""
main.py

Application entry point for the machine learning pipeline.

Runs example experiments using supported datasets and models.
Each experiment creates a dataset, runs multiple machine learning
models through ExperimentRunner, and displays evaluation results.

This file is responsible only for orchestrating experiments.
The implementation details are handled by:
    - datasets
    - models
    - pipeline
    - evaluation
"""




from models.model_type import ModelType
from datasets.titanic import TitanicDataset
from datasets.california_housing import CaliforniaHousingDataset
from pipeline.experiment_runner import ExperimentRunner




def run_titanic() -> dict[str, int]:
    print("=" * 50)
    print("TITANIC SURVIVAL")
    print("=" * 50)
    print()

    # The dataset knows how to Load and preprocess itself.
    # The pipeline doesn't care if this is Titanic, Housing, etc.    
    dataset = TitanicDataset(
        "data/titanic/train.csv"
    )

    # This is done by the ExperimentRunner now
    # dataset.load()
    # dataset.preprocess()
    # dataset.validate()
    # dataset.describe()

    model_types = [
        ModelType.LOGISTIC_REGRESSION_CLASSIFIER,
        ModelType.RANDOM_FOREST_CLASSIFIER
    ]

    experiment_runner = ExperimentRunner(
        dataset,
        model_types
    )

    results = experiment_runner.run()
    
    results.show_comparison()
    results.show_feature_analysis()

    results.compare_predictions(
        model_types[0],
        model_types[1]
    )

    return {
        "datasets_processed": 1,
        "models_evaluated": 2
    }




def run_california_housing() -> dict[str, int]:
    print("=" * 50)
    print("CALIFORNIA HOUSING")
    print("=" * 50)
    print()

    dataset = CaliforniaHousingDataset()

    # This is done by the ExperimentRunner now
    # dataset.load()
    # dataset.preprocess()
    # dataset.validate()
    # dataset.describe()

    model_types = [
        ModelType.LINEAR_REGRESSION,
        ModelType.RANDOM_FOREST_REGRESSOR
    ]

    experiment_runner = ExperimentRunner(
        dataset,
        model_types
    )

    results = experiment_runner.run()

    results.show_comparison()
    results.show_feature_analysis()

    results.compare_predictions(
        model_types[0],
        model_types[1]
    )

    return {
        "datasets_processed": 1,
        "models_evaluated": 2
    }




def main() -> None:

    """
    Runs all example machine learning experiments
    """

    statistics = {
        "datasets_processed": 0,
        "models_evaluated": 0
    }

    for experiment in [
        run_california_housing(),
        run_titanic()
    ]:
        statistics["datasets_processed"] += (
            experiment["datasets_processed"]
        )

        statistics["models_evaluated"] += (
            experiment["models_evaluated"]
        )

    print("=" * 50)
    print("EXPERIMENT COMPLETE")
    print("=" * 50)
    print()

    print(f"Datasets processed : {statistics['datasets_processed']}")
    print(f"Models evaluated   : {statistics['models_evaluated']}")




if __name__ == "__main__":
    main()
