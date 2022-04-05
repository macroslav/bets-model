import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from typing import List, Dict, Tuple, NoReturn


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
        self.train_data = context['train']
        self.test_data = context['test']
        self.cat_features = context['cat_features']
        self.num_features = context['num_features']

        self.val_data = None

    def run_logic(self):

        self._fill_nans()

        self.train_data['target'] = self.train_data.apply(_set_target, axis=1)
        self.test_data['target'] = self.test_data.apply(_set_target, axis=1)

        self._team_encoding()
        self.train_data = self._generate_features(self.train_data)

        self._categorical_encoding()

        self._drop()

        self._split()

        return self.train_data, self.val_data, self.test_data, self.decode_teams, self.teams_labels

    def _fill_nans(self) -> NoReturn:
        pass

    def _drop(self):
        self.train_data = self.train_data.drop(columns=['home_scored', 'away_scored'])

    def _categorical_encoding(self) -> NoReturn:
        """ Encode all cat features with LabelEncoder """
        label_encoder = LabelEncoder()
        for col in self.cat_features:
            self.train_data.loc[:, col] = label_encoder.fit_transform(self.train_data.loc[:, col])
            self.test_data.loc[:, col] = label_encoder.fit_transform(self.test_data.loc[:, col])

    def _split(self) -> NoReturn:

        self.test_data = self.train_data.tail(20)
        drop_index = self.test_data.index
        self.train_data = self.train_data.drop(drop_index, axis=0)

        self.val_data = self.train_data.tail(50)
        drop_index = self.val_data.index
        self.train_data = self.train_data.drop(drop_index, axis=0)

    def _team_encoding(self) -> NoReturn:
        all_teams = self.train_data.sort_values(by='home_team').home_team.unique()
        self.teams_labels = {team: number for number, team in enumerate(all_teams)}
        self.decode_teams = {number: team for number, team in enumerate(all_teams)}

        self.train_data['home_team'] = self.train_data['home_team'].map(self.teams_labels)
        self.train_data['away_team'] = self.train_data['away_team'].map(self.teams_labels)

        self.test_data['home_team'] = self.test_data['home_team'].map(self.teams_labels)
        self.test_data['away_team'] = self.test_data['away_team'].map(self.teams_labels)

    def _generate_features(self, data) -> pd.DataFrame:
        # self._season_averages()
        result = self._current_position_and_win_streaks(data)
        # self._alltime_averages()
        # self._season_totals()
        return result

    def _season_averages(self) -> NoReturn:
        pass

    def _season_totals(self) -> NoReturn:
        pass

    def _current_position_and_win_streaks(self, data) -> NoReturn:

        query = '((home_team == @team) | (away_team == @team)) & (league == @season)'

        data_with_current_points = data.copy()

        for season in data_with_current_points.league.unique():

            for team in data_with_current_points.home_team.unique():

                current_points = 0
                current_win_streak = 0
                current_lose_streak = 0

                team_season_data = data_with_current_points.query(query)

                for idx in team_season_data.index:

                    if team_season_data.loc[idx, 'home_team'] == team:

                        data_with_current_points.loc[idx, 'home_current_points'] = current_points

                        current_points += team_season_data.loc[idx, 'target']

                        data_with_current_points.loc[idx, 'home_current_lose_streak'] = current_lose_streak

                        data_with_current_points.loc[idx, 'home_current_win_streak'] = current_win_streak

                        current_lose_streak = self._calculate_lose_streak(current_lose_streak,
                                                                          team_season_data.loc[idx, 'target'])

                        current_win_streak = self._calculate_win_streak(current_win_streak,
                                                                        team_season_data.loc[idx, 'target'])

                    else:

                        data_with_current_points.loc[idx, 'away_current_points'] = current_points

                        home = team_season_data.loc[idx, 'home_scored']
                        away = team_season_data.loc[idx, 'away_scored']

                        away_match_score = 3 if home < away else 1 if home == away else 0

                        current_points += away_match_score

                        data_with_current_points.loc[idx, 'away_current_lose_streak'] = current_lose_streak

                        data_with_current_points.loc[idx, 'away_current_win_streak'] = current_win_streak

                        current_lose_streak = self._calculate_lose_streak(current_lose_streak, away_match_score)

                        current_win_streak = self._calculate_win_streak(current_win_streak, away_match_score)

        data_with_current_points.home_current_points = data_with_current_points.home_current_points.astype(int)
        data_with_current_points.away_current_points = data_with_current_points.away_current_points.astype(int)
        data_with_current_points.away_current_win_streak = data_with_current_points.away_current_win_streak.astype(
            int)
        data_with_current_points.away_current_lose_streak = data_with_current_points.away_current_lose_streak.astype(
            int)
        data_with_current_points.home_current_win_streak = data_with_current_points.home_current_win_streak.astype(
            int)
        data_with_current_points.home_current_lose_streak = data_with_current_points.home_current_lose_streak.astype(
            int)

        result = self.train_data.merge(data_with_current_points, how='left')

        return result

    def _win_lose_streak(self) -> NoReturn:
        pass

    def _calculate_win_streak(self, actual_win_streak: int, match_result: int) -> int:

        new_win_streak = actual_win_streak

        if match_result == 3:

            new_win_streak += 1

        else:

            new_win_streak = 0

        return new_win_streak

    def _calculate_lose_streak(self, actual_lose_streak: int, match_result: int) -> int:

        new_lose_streak = actual_lose_streak

        if match_result == 0:

            new_lose_streak += 1

        else:

            new_lose_streak = 0

        return new_lose_streak

    def _alltime_averages(self) -> NoReturn:
        pass

    def _max_results(self) -> NoReturn:
        pass
