"""
problem_type.py

Defines the ProblemType enumeration used to identify the type of
machine learning problem being solved.

ProblemType is used to distinguish between classification and
regression problems and allows the application to select the
appropriate models and evaluation logic.

Used by:
    Model
    ModelFactory
    EvaluatorFactory
    ExperimentRunner
"""




from enum import Enum




class ProblemType(Enum):

    CLASSIFICATION = "classification"

    REGRESSION = "regression"
