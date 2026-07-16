from sklearn.linear_model import LinearRegression

from models.model import Model
from models.problem_type import ProblemType




class LinearRegressionModel(Model):

    problem_type = ProblemType.REGRESSION
    

    def __init__(self):
        self.model = LinearRegression()

    
    def train(self, X_train, y_train):
        
        # Teach the model the relationship between
        # the house features and the house price.
        self.model.fit(
            X_train,
            y_train
        )


    def predict(self, X_test):

        # Return predicted house values.
        return self.model.predict(
            X_test
        )
    

    def get_feature_analysis(self):
        return self.model.coef_
    