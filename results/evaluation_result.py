
# Currently deals with Classification, see regression_evaluation_result.py for Regression. Will probably change this to be an abstract class that other sub-classes inherit their behaviour from in the future if we expand more.
class EvaluationResult:

    def __init__(
            self,
            accuracy,
            confusion_matrix
    ):
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix


    # for debugging only. Usage example: print(evaluation)
    def __str__(self):

        return (
            f"Accuracy: {self.accuracy:.2%}\n\n"
            f"Confusion Matrix:\n"
            f"{self.confusion_matrix}"
        )
    

    def show_confusion_matrix(self):
        true_negative = self.confusion_matrix[0][0]
        false_positive = self.confusion_matrix[0][1]
        false_negative = self.confusion_matrix[1][0]
        true_positive = self.confusion_matrix[1][1]

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

    
    def show_errors(self):
        true_negative = self.confusion_matrix[0][0]
        false_positive = self.confusion_matrix[0][1]
        false_negative = self.confusion_matrix[1][0]
        true_positive = self.confusion_matrix[1][1]

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


    def show_report(self):
        print(
            f"Accuracy: {self.accuracy:.2%}"
        )

        print()
        self.show_confusion_matrix()
        print()
        self.show_errors()


    def get_score(self):
        return self.accuracy
    

    def higher_is_better(self):
        return True
    