import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder

from typing import Dict, Tuple, NoReturn, Union, Any
import logging

from feature_generator import FeatureGenerator

TOTAL_THRESHOLD = 2.5


def _remove_by_key(original_dict: Dict, key: str):
    changed_dict = original_dict.copy()
    del changed_dict[key]
    return changed_dict


def _set_target(row) -> int:

    if row.home_scored > row.away_scored:
        return 3
    elif row.home_scored == row.away_scored:
        return 1
    return 0


def _set_total_target(row) -> int:

    if row.home_scored + row.away_scored > 2.5:
        return 1

    return 0


def _set_both_target(row) -> int:

    if row.home_scored > 0 and row.away_scored > 0:
        return 1

    return 0


class DataTransformer:

    def __init__(self, context: Dict):
        self.train_data = context['data']
        self.features = context['features']

        self.transformed_data = self.train_data.copy()
        self.val_data = None

        self.cat_features = self.features['cat_features']

        self.grouped_features = self.features['grouped_features']

        self.team_names = self.grouped_features['names']['team_names']
        self.country_names = self.grouped_features['names']['country_names']
        self.city_names = self.grouped_features['names']['city_names']
        self.manager_names = self.grouped_features['names']['manager_names']

        self.encode_labels = dict()
        self.decode_labels = dict()

    def run_logic(self):

        self._fill_nans()

        self.transformed_data['result_target'] = self.transformed_data.apply(_set_target, axis=1)
        self.transformed_data['total_target'] = self.transformed_data.apply(_set_total_target, axis=1)
        self.transformed_data['both_target'] = self.transformed_data.apply(_set_both_target, axis=1)

        self._names_encoding()
        self._categorical_encoding()
        logging.debug('All categorical features are already encoded!')
        self._generate_features()
        logging.debug('New features are generated!')
        self._drop()

        self._split()

        return self.transformed_data, self.decode_labels

    def run_future_logic(self):

        self._fill_nans()

        self._names_encoding()
        self._categorical_encoding()
        print('All categorical features are already encoded!')
        self._generate_features()
        print('Features are already generated!')

        self._split()

        return self.train_data

    def _fill_nans(self) -> NoReturn:

        self.transformed_data = self.transformed_data.fillna(0)

    def _drop(self):
        self.transformed_data = self.transformed_data.drop(columns=self.grouped_features['scored_features'])

    def _categorical_encoding(self) -> NoReturn:
        """ Encode all cat features with LabelEncoder """
        label_encoder = LabelEncoder()
        for col in self.cat_features:
            if col not in ['country', 'home_manager_country', 'away_manager_country', 'league']:
                try:
                    self.transformed_data.loc[:, col] = label_encoder.fit_transform(self.transformed_data.loc[:, col])
                except Exception:
                    pass
                # self.test_data.loc[:, col] = label_encoder.fit_transform(self.test_data.loc[:, col])

    def _split(self) -> NoReturn:

        self.train_data = self.transformed_data

    def _names_encoding(self) -> NoReturn:

        names = _remove_by_key(self.grouped_features['names'], 'country_names')

        for col_name, (first_col, second_col) in names.items():
            all_values = list(self.transformed_data.sort_values(by=first_col)[first_col].unique())
            all_values += list(self.transformed_data.sort_values(by=second_col)[second_col].unique())
            unique_values = set(all_values)

            self.encode_labels[col_name] = {value: number for number, value in enumerate(unique_values)}
            self.decode_labels[col_name] = {number: value for number, value in enumerate(unique_values)}

            self.transformed_data[first_col] = self.transformed_data[first_col].map(self.encode_labels[col_name])
            self.transformed_data[second_col] = self.transformed_data[second_col].map(self.encode_labels[col_name])

            # self.test_data['home_team'] = self.test_data['home_team'].map(self.encode_labels)
            # self.test_data['away_team'] = self.test_data['away_team'].map(self.encode_labels)

        country, home_manager, away_manager = self.country_names
        all_countries = list(self.transformed_data.sort_values(by=country)[country].unique())
        all_countries += list(self.transformed_data.sort_values(by=home_manager)[home_manager].unique())
        all_countries += list(self.transformed_data.sort_values(by=away_manager)[away_manager].unique())
        unique_countries = set(all_countries)

        leagues = list(self.transformed_data.sort_values(by='league')['league'].unique())

        self.encode_labels['country_names'] = {value: number for number, value in enumerate(unique_countries)}
        self.decode_labels['country_names'] = {number: value for number, value in enumerate(unique_countries)}
        self.encode_labels['leagues'] = {value: number for number, value in enumerate(leagues)}
        self.decode_labels['leagues'] = {number: value for number, value in enumerate(leagues)}

        self.transformed_data[country] = self.transformed_data[country].map(self.encode_labels['country_names'])
        self.transformed_data[home_manager] = self.transformed_data[home_manager].map(
            self.encode_labels['country_names'])
        self.transformed_data[away_manager] = self.transformed_data[away_manager].map(
            self.encode_labels['country_names'])
        self.transformed_data['league'] = self.transformed_data['league'].map(self.encode_labels['leagues'])

    def _generate_features(self) -> NoReturn:

        self.feature_generator = FeatureGenerator(self.transformed_data, self.features)

        self.transformed_data = self.feature_generator.run_generator()

    @staticmethod
    def _set_result_target(self, row) -> int:
        """ Set target feature from score """

        if row.home_scored > row.away_scored:
            return 3
        elif row.home_scored == row.away_scored:
            return 1
        else:
            return 0

    @staticmethod
    def _set_total_target(self, row) -> int:

        if row.home_scored + row.away_scored > TOTAL_THRESHOLD:
            return 1

        return 0


class MatchesToTeamsTransformer:
    """
    This class implements data transformer from matches representation to teams representation

    """

    def __init__(self, train_data: pd.DataFrame, test_data: Union[None, pd.DataFrame]):

        self.train_data = train_data
        self.test_data = test_data

        self.transformed_train = pd.DataFrame()
        self.transformed_test = pd.DataFrame()

    def transform(self) -> Tuple[Union[Union[DataFrame, Series], Any], Union[Union[DataFrame, Series], Any]]:
        self.transformed_test = None
        if isinstance(self.test_data, pd.DataFrame):
            self.transformed_test = self.transform_dataset(self.test_data, 'test')
        self.transformed_train = self.transform_dataset(self.train_data, 'train')

        return self.transformed_train, self.transformed_test

    def transform_dataset(self, data: pd.DataFrame, dataset: str = 'train'):

        first_columns = list(data.iloc[0:1, 3:7].columns)
        first_columns_indexes = [i for i in range(3, 7)]
        second_columns = list(data.iloc[0:1, 1:3].columns)
        second_columns_indexes = [i for i in range(1, 3)]
        third_columns = list(data.iloc[0:1, 9:].columns)
        third_columns_indexes = [i for i in range(9, data.shape[1])]
        transformed_data = pd.DataFrame()

        for index, row in enumerate(data.values):
            temp_data = pd.DataFrame()

            temp_data.loc[0, 'date'] = row[0]
            temp_data.loc[1, 'date'] = row[0]

            temp_data.loc[0, 'team'] = row[7]
            temp_data.loc[1, 'team'] = row[8]
            temp_data.loc[0, 'home'] = 1
            temp_data.loc[1, 'home'] = 0

            free_index = 3
            if dataset == 'train':
                temp_data.loc[0, 'score'] = row[127]
                temp_data.loc[1, 'score'] = row[128]
                free_index = 4

            for col, col_index in zip(first_columns, first_columns_indexes):
                temp_data.insert(free_index, f"{col}", row[col_index], True)
                free_index += 1

            for col, col_index in zip(second_columns, second_columns_indexes):
                temp_data.insert(free_index, f"{col}", row[col_index], True)
                free_index += 1

            for col, col_index in zip(third_columns, third_columns_indexes):
                temp_data.insert(free_index, f"{col}", row[col_index], True)
                free_index += 1

            transformed_data = pd.concat([transformed_data, temp_data], axis=0, ignore_index=True)

        return transformed_data
