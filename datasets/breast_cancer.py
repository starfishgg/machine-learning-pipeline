from sklearn.datasets import fetch_california_housing
from datasets.dataset import Dataset



class CaliforniaHousingDataset(Dataset):
    

    def load(self):
        housing = fetch_california_housing(
            as_frame=True
        )

        self.df = housing.frame
        