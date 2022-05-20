import pandas as pd
from numpy import int64
from configs.paths import TRAIN_DATA_PATH, TEST_DATA_PATH
from data_transformer import MatchesToTeamsTransformer


class DataLoader:
    """ Load train or test data """

    def __init__(self, mode: str = 'fit', data_format: str = 'matches', train_path=TRAIN_DATA_PATH, test_path=None):

        self.data_format = data_format
        self.mode = mode

        self.test_path = test_path
        self.train_path = train_path

        self.train_data = None
        self.test_data = None

    def load_data(self):

        if self.mode == 'predict' and self.test_path:
            self.test_data = self.load_matches(self.test_path)
            self.test_data = self.base_preprocess(self.test_data)

        self.train_data = self.load_matches(self.train_path)
        self.train_data = self.base_preprocess(self.train_data)

        if self.data_format == 'teams':
            self.train_data, self.test_data = MatchesToTeamsTransformer(self.train_data,
                                                                        self.test_data).transform()

        return self.train_data, self.test_data

    @staticmethod
    def load_matches(path):
        data = pd.read_csv(path)
        return data

    @staticmethod
    def base_preprocess(data):

        preprocessed_data = data.copy()

        preprocessed_data['day'] = pd.to_datetime(preprocessed_data.date, format='%d.%m.%Y')
        preprocessed_data['day_of_week'] = preprocessed_data['day'].dt.day_name()
        preprocessed_data['year'] = preprocessed_data['day'].dt.year
        preprocessed_data.date = preprocessed_data.day.values.astype(int64) // 10 ** 9

        preprocessed_data = preprocessed_data.sort_values(by='date')
        preprocessed_data = preprocessed_data.drop(columns=['link'])

        preprocessed_data = preprocessed_data.reset_index()
        preprocessed_data = preprocessed_data.drop(columns=['index'])

        return preprocessed_data
