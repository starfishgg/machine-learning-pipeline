from sklearn.ensemble import RandomForestClassifier

from models.model import Model
from models.problem_type import ProblemType




class RandomForestClassifierModel(Model):

    problem_type = ProblemType.CLASSIFICATION


    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )


    def train(self, X_train, y_train):
        self.model.fit(
            X_train,
            y_train
        )


    def predict(self, X_test):
        return self.model.predict(
            X_test
        )
    

    def get_feature_analysis(self):

        # Random Forest calculates importance by looking at
        # how much each feature helps the trees make decisions.
        return self.model.feature_importances_