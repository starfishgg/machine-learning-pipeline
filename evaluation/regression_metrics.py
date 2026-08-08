"""
regression_metrics.py

Defines the RegressionMetrics class.

RegressionMetrics stores the results produced when evaluating
a regression model.

Contains:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R² score

This class does not perform evaluation itself.
It only stores and presents the results calculated by
RegressionEvaluator.

Created by:
    RegressionEvaluator

Used by:
    Model comparison and reporting code
"""




class RegressionMetrics:

    def __init__(
            self,
            mae: float,
            rmse: float,
            r2: float
    ) -> None:
        self.mae = mae
        self.rmse = rmse
        self.r2 = r2

    
    def __str__(self) -> str:

        return (
            f"MAE: {self.mae:.3f}\n"
            f"RMSE: {self.rmse:.3f}\n"
            f"R²: {self.r2:.3f}"
        )
    

    def show_report(self) -> None:

        print(
            f"MAE: {self.mae:.3f}"
        )

        print(
            f"RMSE: {self.rmse:.3f}"
        )

        print(
            f"R²: {self.r2:.3f}"
        )

    
    def get_score(self) -> float:

        # Lower RMSE means better predictions
        return self.rmse
    

    # Some metrics improve when they increase
    # (accuracy, R²), for others lower is better (RMSE)
    def higher_is_better(self) -> bool:
        return False
    