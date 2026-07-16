from models.problem_type import ProblemType

from evaluation.classification_evaluator import ClassificationEvaluator
from evaluation.regression_evaluator import RegressionEvaluator




class EvaluatorFactory:


    @staticmethod
    def create(problem_type):

        # Choose the correct evaluator depending
        # on the type of ML problem we are solving.
        if problem_type == ProblemType.CLASSIFICATION:
            return ClassificationEvaluator()
        
        if problem_type == ProblemType.REGRESSION:
            return RegressionEvaluator()
        
        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )
    