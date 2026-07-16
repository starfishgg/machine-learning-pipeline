class RegressionEvaluationResult:

    def __init__(
            self,
            mae,
            rmse,
            r2
    ):
        self.mae = mae
        self.rmse = rmse
        self.r2 = r2

    
    def __str__(self):

        return (
            f"MAE: {self.mae:.3f}\n"
            f"RMSE: {self.mae:.3f}\n"
            f"R²: {self.r2:.3f}"
        )
    

    def show_report(self):

        print(
            f"MAE: {self.mae:.3f}"
        )

        print(
            f"RMSE: {self.rmse:.3f}"
        )

        print(
            f"R²: {self.r2:.3f}"
        )

    
    def get_score(self):

        # Lower RMSE means better predictions
        return self.rmse
    

    def higher_is_better(self):
        return False
    