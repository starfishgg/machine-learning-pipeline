"""
titanic.py

Defines the TitanicDataset class, which provides the data loading and
preprocessing logic for the Titanic survival classification dataset.

TitanicDataset inherits the common dataset interface from Dataset and
implements the dataset-specific logic required to prepare the Titanic
data for a classification model.

The target variable represents whether a passenger survived.

Inherits from:
    Dataset

Used by:
    MachineLearningPipeline
    ExperimentRunner
"""




import pandas as pd

from datasets.dataset import Dataset




class TitanicDataset(Dataset):

    DEFAULT_PATH = "data/titanic/train.csv"

    def __init__(self, filepath: str | None = None) -> None:

        super().__init__(filepath)

        self.features = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked_C",
            "Embarked_Q",
            "Embarked_S"
        ]

        self.target = "Survived"

    
    def load(self) -> None:
        self.df = pd.read_csv(self.filepath)

    
    def preprocess(self) -> None:

        self.df["Age"] = (
            self.df["Age"]
            .fillna(self.df["Age"].median())
        )

        self.df["Embarked"] = (
            self.df["Embarked"]
            .fillna(self.df["Embarked"].mode()[0])
        )

        self.df = self.df.drop(
            columns=["Cabin"]
        )

        self.df["Sex"] = self.df["Sex"].map(
            {
                "male": 0,
                "female": 1
            }
        )

        self.df = pd.get_dummies(
            self.df,
            columns=["Embarked"]
        )
