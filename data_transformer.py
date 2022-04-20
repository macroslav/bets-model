import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from typing import List, Dict, Tuple, NoReturn

from feature_generator import FeatureGenerator


def _remove_by_key(original_dict: Dict, key: str):
    changed_dict = original_dict.copy()
    del changed_dict[key]
    return changed_dict


def _set_target(row) -> NoReturn:
    """ Set target feature from score """

    if row.home_scored > row.away_scored:
        return 3
    elif row.home_scored == row.away_scored:
        return 1
    else:
        return 0


class DataTransformer:

    def __init__(self, context: Dict):
        self.raw_data = context['data']
        self.features = context['features']

        self.transformed_data = self.raw_data.copy()
        self.train_data = None
        self.val_data = None
        self.test_data = None

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

        self.transformed_data['target'] = self.transformed_data.apply(_set_target, axis=1)

        self._names_encoding()
        self._categorical_encoding()
        print('All categorical features are already encoded!')
        self._generate_features()
        print('Features are already generated!')
        self._drop()

        self._split()

        return self.train_data, self.val_data, self.test_data, self.decode_labels, self.encode_labels

    def _fill_nans(self) -> NoReturn:

        self.transformed_data = self.transformed_data.fillna(0)

    def _drop(self):
        self.transformed_data = self.transformed_data.drop(columns=self.grouped_features['scored_features'])

    def _categorical_encoding(self) -> NoReturn:
        """ Encode all cat features with LabelEncoder """
        label_encoder = LabelEncoder()
        for col in self.cat_features:
            self.transformed_data.loc[:, col] = label_encoder.fit_transform(self.transformed_data.loc[:, col])
            # self.test_data.loc[:, col] = label_encoder.fit_transform(self.test_data.loc[:, col])

    def _split(self) -> NoReturn:

        self.test_data = self.transformed_data.tail(20)
        drop_index = self.test_data.index
        self.train_data = self.transformed_data.drop(drop_index, axis=0)

        self.val_data = self.train_data.tail(50)
        drop_index = self.val_data.index
        self.train_data = self.train_data.drop(drop_index, axis=0)

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

        team, home_manager, away_manager = self.country_names
        all_countries = list(self.transformed_data.sort_values(by=team)[team].unique())
        all_countries += list(self.transformed_data.sort_values(by=home_manager)[home_manager].unique())
        all_countries += list(self.transformed_data.sort_values(by=away_manager)[away_manager].unique())
        unique_countries = set(all_countries)

        self.encode_labels['country_names'] = {value: number for number, value in enumerate(unique_countries)}
        self.decode_labels['country_names'] = {number: value for number, value in enumerate(unique_countries)}

        self.transformed_data[team] = self.transformed_data[team].map(self.encode_labels['country_names'])
        self.transformed_data[home_manager] = self.transformed_data[home_manager].map(
            self.encode_labels['country_names'])
        self.transformed_data[away_manager] = self.transformed_data[away_manager].map(
            self.encode_labels['country_names'])

    def _generate_features(self) -> NoReturn:

        self.feature_generator = FeatureGenerator(self.transformed_data, self.features)

        self.transformed_data = self.feature_generator.run_generator()

    # def _season_averages(self) -> NoReturn:
    #     pass
    #
    # def _season_totals(self) -> NoReturn:
    #     pass
    #
    # def _cumulative_features(self, data) -> NoReturn:
    #
    #     query = '((home_team == @team) | (away_team == @team)) & (league == @season)'
    #
    #     data_with_current_points = data.copy()
    #
    #     for season in data_with_current_points.league.unique():
    #
    #         for team in data_with_current_points.home_team.unique():
    #
    #             current_points = 0
    #             current_win_streak = 0
    #             current_lose_streak = 0
    #
    #             team_season_data = data_with_current_points.query(query)
    #
    #             for idx in team_season_data.index:
    #
    #                 if team_season_data.loc[idx, 'home_team'] == team:
    #
    #                     data_with_current_points.loc[idx, 'current_home_points'] = current_points
    #
    #                     current_points += team_season_data.loc[idx, 'target']
    #
    #                     data_with_current_points.loc[idx, 'current_home_lose_streak'] = current_lose_streak
    #
    #                     data_with_current_points.loc[idx, 'current_home_win_streak'] = current_win_streak
    #
    #                     current_lose_streak = self._calculate_lose_streak(current_lose_streak,
    #                                                                       team_season_data.loc[idx, 'target'])
    #
    #                     current_win_streak = self._calculate_win_streak(current_win_streak,
    #                                                                     team_season_data.loc[idx, 'target'])
    #
    #                 else:
    #
    #                     data_with_current_points.loc[idx, 'current_away_points'] = current_points
    #
    #                     home = team_season_data.loc[idx, 'home_scored']
    #                     away = team_season_data.loc[idx, 'away_scored']
    #
    #                     away_match_score = 3 if home < away else 1 if home == away else 0
    #
    #                     current_points += away_match_score
    #
    #                     data_with_current_points.loc[idx, 'current_away_lose_streak'] = current_lose_streak
    #
    #                     data_with_current_points.loc[idx, 'current_away_win_streak'] = current_win_streak
    #
    #                     current_lose_streak = self._calculate_lose_streak(current_lose_streak, away_match_score)
    #
    #                     current_win_streak = self._calculate_win_streak(current_win_streak, away_match_score)
    #
    #     data_with_current_points.home_current_points = data_with_current_points.home_current_points.astype(int)
    #     data_with_current_points.away_current_points = data_with_current_points.away_current_points.astype(int)
    #     data_with_current_points.away_current_win_streak = data_with_current_points.away_current_win_streak.astype(
    #         int)
    #     data_with_current_points.away_current_lose_streak = data_with_current_points.away_current_lose_streak.astype(
    #         int)
    #     data_with_current_points.home_current_win_streak = data_with_current_points.home_current_win_streak.astype(
    #         int)
    #     data_with_current_points.home_current_lose_streak = data_with_current_points.home_current_lose_streak.astype(
    #         int)
    #
    #     result = self.train_data.merge(data_with_current_points, how='left')
    #
    #     return result
    #
    # def _win_lose_streak(self) -> NoReturn:
    #     pass
    #
    # def _calculate_win_streak(self, actual_win_streak: int, match_result: int) -> int:
    #
    #     new_win_streak = actual_win_streak
    #
    #     if match_result == 3:
    #
    #         new_win_streak += 1
    #
    #     else:
    #
    #         new_win_streak = 0
    #
    #     return new_win_streak
    #
    # def _calculate_lose_streak(self, actual_lose_streak: int, match_result: int) -> int:
    #
    #     new_lose_streak = actual_lose_streak
    #
    #     if match_result == 0:
    #
    #         new_lose_streak += 1
    #
    #     else:
    #
    #         new_lose_streak = 0
    #
    #     return new_lose_streak
    #
    # def _alltime_averages(self) -> NoReturn:
    #     pass
    #
    # def _max_results(self) -> NoReturn:
    #     pass
