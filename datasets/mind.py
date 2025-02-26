import pandas as pd

from datasets.base import AbstractDataset


class MINDDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return "MIND"

    @classmethod
    def url(cls):
        return "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_train.zip"

    @classmethod
    def zip_file_content_is_folder(cls):
        return False

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings_full.csv')
        df = pd.read_csv(file_path, sep=',', header=0)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

    def make_implicit(self, df):
        # Dataset is already implicit
        return df