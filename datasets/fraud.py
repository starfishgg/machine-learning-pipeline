"""
fraud.py

Defines the FraudDataset class used by the machine learning pipeline.

FraudDataset loads and preprocesses a large financial transaction dataset
containing over 6 million transactions. The dataset includes transaction
types, transaction amounts, account balances, and fraud labels.

Target:
    isFraud
        Indicates whether a transaction is fraudulent.

Existing detection rule:
    isFlaggedFraud
        Indicates whether the transaction was flagged by the dataset's
        existing fraud-detection rule. This is retained for baseline
        evaluation but excluded from the machine learning features.

Feature engineering:
    Balance-based features are derived from the relationship between
    transaction amounts and account balances. These features are intended
    to capture inconsistencies in the movement of money between accounts.

Categorical data:
    The transaction type is one-hot encoded so it can be used by
    scikit-learn models.

Excluded identifiers:
    nameOrig and nameDest are transaction/account identifiers and are
    excluded from the model rather than treated as categorical features.

The resulting dataset can be used by the pipeline to compare different
fraud classification models, including Logistic Regression and Random
Forest.

Using the dataset at: https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

"""




import pandas as pd

from datasets.dataset import Dataset




class FraudDataset(Dataset):

    DEFAULT_PATH: str = "data/fraud/AIML Dataset.csv"


    def __init__(self, filepath: str | None = None) -> None:

        super().__init__(filepath)

        self.target = "isFraud"


    def load(self) -> None:
        self.df = pd.read_csv(self.filepath)


    def preprocess(self) -> None:

        # checks for errors in the origin balance, which is the difference between the old balance minus the amount and the new balance.
        self.df["origin_balance_change"] = (
            self.df["oldbalanceOrg"] - self.df["newbalanceOrig"]
        )

        # checks for errors in the destination balance, which is the difference between the old balance plus the amount and the new balance.
        self.df["destination_balance_change"] = (
            self.df["newbalanceDest"] - self.df["oldbalanceDest"]
        )

        # errors in the origin balance, which is the difference between the old balance minus the amount and the new balance.
        self.df["origin_balance_error"] = (
            self.df["oldbalanceOrg"] 
            - self.df["amount"]
            - self.df["newbalanceOrig"]
        )

        # errors in the destination balance, which is the difference between the old balance plus the amount and the new balance.
        self.df["destination_balance_error"] = (
            self.df["oldbalanceDest"] 
            + self.df["amount"]
            - self.df["newbalanceDest"]
        )

        # Convert categorical values into numerical columns so that
        # scikit-learn models can process them.
        self.df = pd.get_dummies(
            self.df,
            columns=["type"],
            dtype=int
        )


        # Remove columns that are identifiers, targets, or raw values
        # that have already been transformed into useful features.
        self.df = self.df.drop(
            columns=[
                "nameOrig",
                "nameDest",
                "isFlaggedFraud"
            ]
        )

        # Build the features list automatically now we have transformed the dataset into a final form that can be used by our models. This is more robust than hard-coding the features list.
        self.features = [
            column
            for column in self.df.columns
            if column != self.target
        ]
