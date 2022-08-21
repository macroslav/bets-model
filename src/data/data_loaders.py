import pandas as pd
from numpy import int64
from src.data.data_transformers import MatchesToTeamsTransformer

from typing import Protocol, Optional
from pathlib import Path


class BaseDataLoader(Protocol):

    def load_data(self):
        """ Load data """

    def base_preprocess(self):
        """ Realize basic processing like setting columns types and sorting """


class DataLoader:
    """ Load train or test data """

    def __init__(self, train_path: Path, future_path: Optional[Path] = None):

        self.train_path = train_path
        self.future_path = future_path

        self.train_data = None
        self.future_data = None

    def __call__(self):
        self.load_data()
        self.train_data = self.base_preprocess(self.train_data)
        if isinstance(self.future_data, pd.DataFrame):
            self.future_data = self.base_preprocess(self.future_data)
        return self.train_data, self.future_data

    def load_data(self):
        self.train_data = self._load_matches(self.train_path)
        if isinstance(self.future_path, Path):
            self.future_data = self._load_matches(self.future_data)

    @staticmethod
    def _load_matches(path):
        data = pd.read_csv(path)
        return data

    @staticmethod
    def base_preprocess(data):

        preprocessed_data = data.copy()

        preprocessed_data['date'] = preprocessed_data['date'].astype(str) + ' ' + preprocessed_data['time']
        preprocessed_data['date'] = pd.to_datetime(preprocessed_data['date'], format='%d.%m.%Y %H:%M')

        columns = data.columns.tolist()

        preprocessed_data['day'] = preprocessed_data.date.dt.date
        preprocessed_data['day_of_week'] = preprocessed_data.date.dt.day_name()
        preprocessed_data['week'] = preprocessed_data.date.dt.isocalendar().week
        preprocessed_data['year'] = preprocessed_data.date.dt.year
        preprocessed_data['timestamp'] = preprocessed_data.date.values.astype(int64) // 10 ** 9

        preprocessed_data['home_manager_age'] /= 365
        preprocessed_data['away_manager_age'] /= 365

        preprocessed_data = preprocessed_data.sort_values(by='date')
        preprocessed_data = preprocessed_data.drop(columns=['link'])

        preprocessed_data.reset_index(inplace=True, drop=True)

        return preprocessed_data
