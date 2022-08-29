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

    def __init__(self, train_dir: Path, future_path: Optional[Path] = None):

        self.train_dir = train_dir
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

        self.train_data = self._load_train()
        self.future_data = self._load_future()

    def _load_train(self) -> pd.DataFrame:
        all_data = pd.DataFrame()
        chunk_paths = Path.iterdir(self.train_dir)
        for chunk_path in chunk_paths:
            data_chunk = pd.read_csv(chunk_path)
            all_data = pd.concat([all_data,
                                  data_chunk], ignore_index=True)
        return all_data

    def _load_future(self) -> Optional[pd.DataFrame]:
        if isinstance(self.future_path, Path):
            return pd.read_csv(self.future_path)

        return None

    @staticmethod
    def base_preprocess(data):

        preprocessed_data = data.copy()

        preprocessed_data['date'] = preprocessed_data['date'].astype(str) + ' ' + preprocessed_data['time']
        preprocessed_data['date'] = pd.to_datetime(preprocessed_data['date'], format='%d.%m.%Y %H:%M')

        preprocessed_data['day'] = preprocessed_data.date.dt.date
        preprocessed_data['day'] = pd.to_datetime(preprocessed_data['day'])
        preprocessed_data['timestamp_date'] = preprocessed_data.day.values.astype(int64) // 10 ** 9

        preprocessed_data['day_of_week'] = preprocessed_data.date.dt.day_name()
        preprocessed_data['week'] = preprocessed_data.date.dt.isocalendar().week
        preprocessed_data['year'] = preprocessed_data.date.dt.year
        preprocessed_data['timestamp_match'] = preprocessed_data.date.values.astype(int64) // 10 ** 9

        #
        # preprocessed_data['home_manager_age'] /= 365
        # preprocessed_data['away_manager_age'] /= 365

        preprocessed_data = preprocessed_data.sort_values(by='date')
        preprocessed_data = preprocessed_data.drop(columns=['link'])

        preprocessed_data.reset_index(inplace=True, drop=True)

        return preprocessed_data
