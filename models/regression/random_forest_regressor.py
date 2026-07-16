from sklearn.ensemble import RandomForestRegressor

from models.model import Model
from models.problem_type import ProblemType




class RandomForestRegressorModel(Model):

    problem_type = ProblemType.REGRESSION
    

    def __init__(self):

        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )


    def train(self, X_train, y_train):

        # Train using known house prices.
        self.model.fit(
            X_train,
            y_train
        )


    def predict(self, X_test):
        
        # Predict continous house values.
        return self.model.predict(
            X_test
        )
    

    def get_feature_analysis(self):

        # Random forest already calculate how much
        # each feature contributed to predictions.
        return self.model.feature_importances_