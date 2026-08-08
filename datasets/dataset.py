"""
dataset.py

Defines the abstract Dataset base class used by the machine learning
pipeline.

Dataset provides the common interface and shared functionality required
by all supported datasets. Concrete dataset classes inherit from this
class and implement their own dataset-specific loading and preprocessing
logic.

Abstract methods:
    load()
        Loads the dataset into the internal DataFrame.

    preprocess()
        Cleans and prepares the dataset for machine learning.

Used/Implementaed by:
    TitanicDataset
    CaliforniaHousingDataset

The common interface allows the machine learning pipeline to work with
different datasets without needing to know how each dataset is loaded
or prepared.
"""




import copy
import pandas as pd
from abc import ABC, abstractmethod




class Dataset(ABC):
    
    DEFAULT_PATH: str = None

    def __init__(self, filepath:str | None = None) -> None:

        # Store where the dataset comes from.
        # Some datasets may use CSV files,
        # others may load from sklearn or an API.
        self.filepath: str | None = filepath or self.DEFAULT_PATH

        # The loaded dataframe.
        self.df: pd.DataFrame | None = None

        # Columns used as model inputs.
        self.features: list[str] = []

        # Column we want to predict.
        self.target: str | None = None


    @abstractmethod
    def load(self) -> None:
        """Load the dataset into self.df"""
        pass


    @abstractmethod
    def preprocess(self) -> None:
        """Clean and prepare the dataset"""
        pass

    
    def get_features(self) -> pd.DataFrame:
        return self.df[self.features]
    
    
    def get_target(self) -> pd.Series:
        return self.df[self.target]
    

    def get_dataframe(self) -> pd.DataFrame:
        # Safer to use a copy so we don't accidentally mutate our dataset.
        return self.df.copy()


    # return type in quotes as we are referring to the clas from inside
    # its own definition.
    def clone(self) -> "Dataset":
        return copy.deepcopy(self)


    def get_feature_names(self) -> list[str]:
        # The model only sees the processed features.
        # Return the names so we can explain the model output later.
        # Safer to use a copy so we don't accidentally mutate our dataset.
        return self.features.copy()
    

    def validate(self) -> None:

        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        
        missing_features = [
            feature
            for feature in self.features
            if feature not in self.df.columns

        ]

        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        if not self.features:
            raise ValueError("No features have been defined")
        
        if self.target is None:
            raise ValueError("No target has been defined.")
        
        if self.target not in self.df.columns:
            raise ValueError(f"Missing target: {self.target}")
        

    def describe(self) -> None:
        print("DATASET")
        print("-------")
        print(f"Samples : {len(self.df):,}")
        print(f"Features: {len(self.features)}")
        print(f"Target  : {self.target}")
        missing = self.df.isnull().sum().sum()
        print(f"Missing values: {missing}")

        print()
