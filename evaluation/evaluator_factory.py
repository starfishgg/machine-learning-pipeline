"""
evaluator_factory.py

Provides a factory for creating the correct Evaluator implementation.

The factory selects an evaluator based on the type of machine
learning problem being solved.

Example:
    Classification problem
        -> ClassificationEvaluator

    Regression problem
        -> RegressionEvaluator

This keeps the rest of the application independent from the
specific evaluator implementation being used.

Created by:
    ExperimentResult

Creates:
    ClassificationEvaluator
    RegressionEvaluator
"""




from models.problem_type import ProblemType
from evaluation.classification_evaluator import ClassificationEvaluator
from evaluation.regression_evaluator import RegressionEvaluator
from evaluation.evaluator import Evaluator




class EvaluatorFactory:


    @staticmethod
    def create(problem_type: ProblemType) -> Evaluator:

        # Choose the correct evaluator depending
        # on the type of ML problem we are solving.
        if problem_type == ProblemType.CLASSIFICATION:
            return ClassificationEvaluator()
        
        if problem_type == ProblemType.REGRESSION:
            return RegressionEvaluator()
        
        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )
    