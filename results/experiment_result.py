from results.evaluation_result import EvaluationResult
from results.evaluator_factory import EvaluatorFactory
from evaluation.classification_evaluator import ClassificationEvaluator
from models.problem_type import ProblemType



class ExperimentResult:

    def __init__(self, results, feature_names):


        # Store all model prediction results.
        #
        # Example:
        #
        # {
        #   ModelType.RANDOM_FOREST_CLASSIFIER: PredictionResult,
        #   ModelType.LOGISTIC_REGRESSION: PredictionResult
        # }
        #
        self.results = results
        self.feature_names = feature_names

    
    def get_models(self):
        return self.results.keys()
    

    def get_result(self, model_type):
        return self.results[model_type]
    

    def show_comparison(self):

        best_model = None
        best_score = None

        print("MODEL COMPARISON")
        print("----------------")
        print()


        for model_type, model_result in self.results.items():

            # Create the correct evaluator based on whether
            # this model is classification or regression.
            evaluator = EvaluatorFactory.create(
                model_result.model.problem_type
            )

            # Evaluate this model.
            evaluation = evaluator.evaluate(
                model_result.get_prediction_result()
            )

            score = evaluation.get_score()


            print(model_type.name)
            evaluation.show_report()

            print()


            # First model automatically becomes the current best.
            if best_score is None:
                best_score = score
                best_model = model_type
            
            else:
                # Some metrics improve when theyh increase
                # (accuracy, R²)
                if evaluation.higher_is_better():
                    if score > best_score:
                        best_score = score
                        best_model = model_type
            
                # Some metrics improve when they decrease
                # (RMSE, MAE).
                else:

                    if score < best_score:
                        best_score = score
                        best_model = model_type

        print("BEST MODEL")
        print("----------")

        print(best_model.name)
        # print(f"Score: {best_score:.3f}")

        print()
    




    def compare_predictions(self, first_model, second_model):

        first_result = (
            self.results[first_model].get_prediction_result()
        )

        second_result = (
            self.results[second_model].get_prediction_result()
        )

        predictions1 = first_result.get_predictions()
        predictions2 = second_result.get_predictions()

        problem_type = (
            self.results[first_model].model.problem_type
        )

        print("MODEL AGREEMENT")
        print("---------------")
        print()

        #print(f"{first_model.name} vs {second_model.name}")

        if problem_type == ProblemType.CLASSIFICATION:
            same = sum(
                a == b
                for a, b in zip(
                    predictions1,
                    predictions2
                )
            )

            total = len(predictions1)

            print(
                f"Same predictions: {same}/{total}"
            )

            print(
                f"Different predictions: {total - same}/{total}"
            )
        
        elif problem_type == ProblemType.REGRESSION:

            differences = [
                abs(a - b)
                for a, b in zip(
                    predictions1,
                    predictions2
                )
            ]

            print(
                f"Mean prediction difference: {sum(differences)/len(differences):.3f}"
            )

            print(
                f"Maximum prediction difference: {max(differences):.3f}"
            )
        print()


    def show_feature_analysis(self):
        
        print("FEATURE ANALYSIS")
        print("----------------")
        print()

        for model_type, model_result in self.results.items():

            print(model_type.name)

            analysis = model_result.get_feature_analysis()

            features_with_values = list(
                zip(
                    self.feature_names,
                    analysis
                )    
            )

            # Sort by absolute value so this works for
            # both Random Forest importance and Logistic Reghression coefficients
            features_with_values.sort(
                key=lambda item: abs(item[1]),
                reverse=True
            )

            for feature, value in features_with_values:
                print(f"{feature}: {value:.4f}")
            print()
