"""
classification_metrics.py

Defines the ClassificationMetrics class.

ClassificationMetrics stores the results produced when evaluating
a classification model.

Contains:
    - Accuracy score
    - Confusion matrix

This class does not perform evaluation itself.
It only stores and presents the results calculated by
ClassificationEvaluator.

Created by:
    ClassificationEvaluator

Used by:
    Model comparison and reporting code
"""




import numpy as np




class ClassificationMetrics:

    def __init__(
            self,
            accuracy_score: float,
            confusion_matrix_values: np.ndarray
    ) -> None:
        
        self.accuracy_score: float = accuracy_score
        self.confusion_matrix_values: np.ndarray = confusion_matrix_values


    # for debugging only. Usage example: print(evaluation)
    def __str__(self) -> str:

        return (
            f"Accuracy: {self.accuracy_score:.2%}\n\n"
            f"Confusion Matrix:\n"
            f"{self.confusion_matrix_values}"
        )
    

    def show_confusion_matrix(self) -> None:

        true_negative = self.confusion_matrix_values[0][0]
        false_positive = self.confusion_matrix_values[0][1]
        false_negative = self.confusion_matrix_values[1][0]
        true_positive = self.confusion_matrix_values[1][1]

        print("CONFUSION MATRIX")
        print("================")
        print()

        print("                     Predicted")
        print("                Negative   Positive")

        print(
            f"Actual Negative   {true_negative:>4}        {false_positive:>4}"
        )

        print(
            f"Actual Positive   {false_negative:>4}        {true_positive:>4}"
        )

    
    def show_errors(self) -> None:
        true_negative = self.confusion_matrix_values[0][0]
        false_positive = self.confusion_matrix_values[0][1]
        false_negative = self.confusion_matrix_values[1][0]
        true_positive = self.confusion_matrix_values[1][1]

        print()
        print("ERROR ANALYSIS")
        print("==============")
        print()

        print(
            f" True Negatives: {true_negative}"
        )
        print(
            f"False Positives: {false_positive}"
        )
        print(
            f"False Negatives: {false_negative}"
        )
        print(
            f" True Positives: {true_positive}"
        )

        total = (
            true_positive +
            true_negative +
            false_positive +
            false_negative
        )

        correct = (
            true_positive +
            true_negative
        )

        print()

        print(
            f"The model correctly classified {correct} of {total} samples."
        )

        print()


    def show_report(self) -> None:
        print(
            f"Accuracy: {self.accuracy_score:.2%}"
        )

        print()
        self.show_confusion_matrix()
        print()
        self.show_errors()


    def get_score(self) -> float:
        return self.accuracy_score

    
    # Some metrics improve when they increase
    # (accuracy, R²)
    def higher_is_better(self) -> bool:
        return True
    