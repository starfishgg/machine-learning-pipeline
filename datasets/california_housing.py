from sklearn.datasets import fetch_california_housing
from datasets.dataset import Dataset



class CaliforniaHousingDataset(Dataset):

    def __init__(self, filepath=None):

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
    

    def load(self):

        # This dataset is loaded from sklearn (as a pandas DataFrame)
        # rather than a CSV file.
        # The base Dataseet still provides filepath support because
        # other datasets may load from files.
        housing = fetch_california_housing(as_frame=True)
        self.df = housing.frame
        
    
    def preprocess(self):

        # The California Housing dataset is already
        # cleaned and numeric, so there isn't anything 
        # to preprocess yet
        pass
