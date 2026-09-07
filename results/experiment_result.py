"""
experiment_result.py

Represents the outcome of a complete machine learning experiment.

An experiment consists of one or more ModelResults, with each
ModelResult containing the trained model and the predictions it
produced.

ExperimentResult provides methods for:
- Accessing individual model results.
- Comparing model performance.
- Comparing predictions between models.
- Analysing feature importance or model coefficients.

Created by:
    ExperimentRunner

Uses:
    ModelResult
    EvaluatorFactory
    ProblemType
"""




from collections.abc import KeysView

from evaluation.evaluator_factory import EvaluatorFactory
from models.problem_type import ProblemType
from models.model_type import ModelType
from results.model_result import ModelResult




class ExperimentResult:

    def __init__(
        self, 
        model_results: dict[ModelType, ModelResult], 
        feature_names: list[str]
    ) -> None:


        # Store the result of each model using ModelType as athey key.
        #
        # Example:
        #
        # {
        #   ModelType.RANDOM_FOREST_CLASSIFIER: PredictionSet,
        #   ModelType.LOGISTIC_REGRESSION: PredictionSet
        # }
        #
        self.model_results = model_results

        # Store the names of the features used by the models.
        # These are needed later when displaying feature analysis.
        self.feature_names = feature_names

    
    def get_models(self) -> KeysView[ModelType]:
        return self.model_results.keys()
    

    def get_model_result(self, model_type: ModelType) -> ModelResult:
        return self.model_results[model_type]
    

    def show_comparison(self) -> None:

        best_model: ModelType | None = None
        best_model_score: float | None = None
        best_models: list[ModelType] = []

        print("MODEL COMPARISON")
        print("----------------")
        print()


        for model_type, model_result in self.model_results.items():

            # Create the correct evaluator based on whether
            # this model is classification or regression.
            evaluator = EvaluatorFactory.create(
                model_result.model.problem_type
            )

            # Evaluate the predictions produced by this model..
            metrics = evaluator.evaluate(
                model_result.get_prediction_set()
            )

            score = metrics.get_score()

            print(model_type.name)
            metrics.show_report()
            print()


            # First model establishes the initial best score.
            if best_model_score is None:
                best_model_score = score
                best_models = [model_type]

            # This model has the same score as the current best.
            elif score == best_model_score:
                best_models.append(model_type)

            # Some metrics improve when they increase
            # (accuracy, R²)            
            elif metrics.higher_is_better():
                if score > best_model_score:
                    best_model_score = score
                    best_models = [model_type]
            
            # Other metrics improve when they decrease
            # (RMSE, MAE).
            else:
                if score < best_model_score:
                    best_model_score = score
                    best_models = [model_type]

        print("BEST MODEL")
        print("----------")

        if len(best_models) == 1:
            print(best_models[0].name)
        else:
            print("Models performed equally:")
            for model_type in best_models:
                print(f"- {model_type.name}")

        print()
    



    # TODO: Perhaps change this in the future to iterate through a list
    # of models rather than just comparing two? We will need to add
    # more models before we do that though...
    def compare_predictions(
        self,
        first_model: ModelType,
        second_model: ModelType
    ) -> None:

        first_prediction_set = (
            self.model_results[first_model].get_prediction_set()
        )

        second_prediction_set = (
            self.model_results[second_model].get_prediction_set()
        )

        first_predictions = first_prediction_set.get_predictions()
        second_predictions = second_prediction_set.get_predictions()

        problem_type = (
            self.model_results[first_model].model.problem_type
        )

        print("MODEL PREDICTION DIFFERENCE")
        print("---------------------------")
        print()

        #print(f"{first_model.name} vs {second_model.name}")

        # Classification models produce discrete predictions (yes/no),
        # so we can count how often the two models agree.
        if problem_type == ProblemType.CLASSIFICATION:
            same_predictions = sum(
                first_prediction == second_prediction
                for first_prediction, second_prediction in zip(
                    first_predictions,
                    second_predictions
                )
            )

            total_predictions = len(first_predictions)

            print(
                f"Same predictions: {same_predictions}/{total_predictions}"
            )

            print(
                f"Different predictions: {total_predictions - same_predictions}/{total_predictions}"
            )

        # Regression models produce continuous values, so instead 
        # ofexact agreement we measure the size of the difference
        # between their predictions.
        elif problem_type == ProblemType.REGRESSION:

            prediction_differences = [
                abs(first_prediction - second_prediction)
                for first_prediction, second_prediction in zip(
                    first_predictions,
                    second_predictions
                )
            ]

            print(
                f"Mean prediction difference: {sum(prediction_differences)/len(prediction_differences):.3f}"
            )

            print(
                f"Maximum prediction difference: {max(prediction_differences):.3f}"
            )
        print()


    def show_feature_analysis(self) -> None:
        
        print("FEATURE ANALYSIS")
        print("----------------")
        print()

        for model_type, model_result in self.model_results.items():

            print(model_type.name)

            feature_analysis = model_result.get_feature_analysis()

            # Pair each feature anme with its corresponding model value.
            features_with_values = list(
                zip(
                    self.feature_names,
                    feature_analysis
                )    
            )

            # Sort by absolute value so this works for both:
            #  Random Forest importance 
            #  Linea/Logistic Reghression coefficients
            features_with_values.sort(
                key=lambda item: abs(item[1]),
                reverse=True
            )

            for feature_name, value in features_with_values:
                print(f"{feature_name}: {value:.4f}")
            print()

        #print("FEATURE NAMES:", len(self.feature_names))
        #print("FEATURE ANALYSIS:", len(feature_analysis))