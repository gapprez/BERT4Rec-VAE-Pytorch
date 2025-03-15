import pandas as pd

from .base import AbstractDataset


class ML100kDataset(AbstractDataset):
    @classmethod
    def url(cls):
        return 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

    @classmethod
    def code(cls):
        return 'ml-latest-small'

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.csv',
                'ratings.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.csv')
        df = pd.read_csv(file_path, header=0)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
