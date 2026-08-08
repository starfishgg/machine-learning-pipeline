"""
california_housing.py

Defines the CaliforniaHousingDataset class, which provides the data
loading and preprocessing logic for the California Housing regression
dataset.

CaliforniaHousingDataset inherits from the abstract Dataset base class
and implements the dataset-specific logic required to prepare the
housing data for a regression model.

The dataset is loaded directly through scikit-learn rather than from a
local file, so no default file path is required.

The target variable represents the median house value.

Inherits from:
    Dataset

Used by:
    MachineLearningPipeline
    ExperimentRunner
"""




from sklearn.datasets import fetch_california_housing
from datasets.dataset import Dataset




class CaliforniaHousingDataset(Dataset):

    def __init__(self, filepath:str | None = None):

        super().__init__(filepath)

        self.features = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]

        self.target = "MedHouseVal"
    

    def load(self) -> None:

        # This dataset is loaded from sklearn (as a pandas DataFrame)
        # rather than a CSV file.
        # The base Dataseet still provides filepath support because
        # other datasets may load from files.
        housing_data = fetch_california_housing(as_frame=True)
        self.df = housing_data.frame
        
    
    def preprocess(self) -> None:

        # The California Housing dataset is already
        # cleaned and numeric, so there isn't anything 
        # to preprocess yet
        pass
