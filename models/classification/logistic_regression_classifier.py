from sklearn.linear_model import LogisticRegression

from models.model import Model
from models.problem_type import ProblemType




class LogisticRegressionClassifierModel(Model):

    problem_type = ProblemType.CLASSIFICATION
    

    def __init__(self):

        # Logistic regression is a simple linear classification model.
        # max_iter is increased bacause the default sometimes isn't enough
        # for the algoritm to converge.
        self.model = LogisticRegression(
            max_iter=1000
        )


    def train(self, X_train, y_train):

        # Learn the relationship between the input and the outcomes
        self.model.fit(
            X_train,
            y_train
        )


    def predict(self, X_test):

        # Predict results the model hasn't seen before.
        return self.model.predict(
            X_test
        )
    
    def get_feature_analysis(self):
        return self.model.coef_[0]