import copy
from abc import ABC, abstractmethod


class Dataset(ABC):
    
    DEFAULT_PATH = None

    def __init__(self, filepath=None):

        # Store where the dataset comes from.
        # Some datasets may use CSV files,
        # others may load from sklearn or an API.
        self.filepath = filepath or self.DEFAULT_PATH

        # The loaded dataframe.
        self.df = None

        # Columns used as model inputs.
        self.features = []

        # Column we want to predict.
        self.target = None

    @abstractmethod
    def load(self):
        """Load the dataset into self.df"""
        pass

    @abstractmethod
    def preprocess(self):
        """Clean and prepare the dataset"""
        pass

    
    def get_features(self):
        return self.df[self.features]
    
    
    def get_target(self):
        return self.df[self.target]
    

    def get_dataframe(self):
        # Safer to use a copy so we don't accidentally mutate our dataset.
        return self.df.copy()

    def clone(self):
        return copy.deepcopy(self)

    def get_feature_names(self):
        # The model only sees the processed features.
        # Return the names so we can explain the model output later.
        # Safer to use a copy so we don't accidentally mutate our dataset.
        return self.features.copy()
    

    def validate(self):

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
        

    def describe(self):
        print("DATASET")
        print("-------")
        print(f"Samples : {len(self.df):,}")
        print(f"Features: {len(self.features)}")
        print(f"Target  : {self.target}")
        missing = self.df.isnull().sum().sum()
        print(f"Missing values: {missing}")

        print()
