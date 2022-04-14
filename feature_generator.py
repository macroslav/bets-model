import pandas as pd
import numpy as np

from typing import List, Tuple, Dict, NoReturn


class FeatureGenerator:
    """ Class to generate new features"""
    def __init__(self,
                 data: pd.DataFrame,
                 grouped_features: Dict[List],
                 cat_features: Tuple,
                 numeric_features: Tuple,
                 money_features: Tuple):

        self.raw_data = data
        self.grouped_features = grouped_features
        self.cat_features = cat_features
        self.numeric_features = numeric_features
        self.money_features = money_features

        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.val_data = pd.DataFrame()

    def alltime_average(self):
        pass

    def season_average(self):
        pass

    def cumulative(self):

        query = '((home_team == @team) | (away_team == @team)) & (league == @season)'

        data_with_current_points = self.raw_data.copy()

        for season in data_with_current_points.league.unique():

            for team in data_with_current_points.home_team.unique():

                current_points = 0
                current_win_streak = 0
                current_lose_streak = 0
                current_scored = 0
                current_missed = 0

                team_season_data = data_with_current_points.query(query)

                for idx in team_season_data.index:

                    if team_season_data.loc[idx, 'home_team'] == team:

                        data_with_current_points.loc[idx, 'current_home_points'] = current_points
                        current_points += team_season_data.loc[idx, 'target']

                        data_with_current_points.loc[idx, 'current_home_scored'] = current_scored
                        current_scored += team_season_data.loc[idx, 'home_scored']

                        data_with_current_points.loc[idx, 'current_home_missed'] = current_missed
                        current_missed += team_season_data.loc[idx, 'away_scored']

                        data_with_current_points.loc[idx, 'current_home_diff'] = current_scored - current_missed

                        data_with_current_points.loc[idx, 'current_home_lose_streak'] = current_lose_streak
                        data_with_current_points.loc[idx, 'current_home_win_streak'] = current_win_streak

                        current_lose_streak = self._calculate_lose_streak(current_lose_streak,
                                                                          team_season_data.loc[idx, 'target'])

                        current_win_streak = self._calculate_win_streak(current_win_streak,
                                                                        team_season_data.loc[idx, 'target'])

                    else:

                        data_with_current_points.loc[idx, 'current_away_points'] = current_points

                        home = team_season_data.loc[idx, 'home_scored']
                        away = team_season_data.loc[idx, 'away_scored']

                        away_match_score = 3 if home < away else 1 if home == away else 0

                        current_points += away_match_score

                        data_with_current_points.loc[idx, 'current_away_lose_streak'] = current_lose_streak
                        data_with_current_points.loc[idx, 'current_away_win_streak'] = current_win_streak

                        data_with_current_points.loc[idx, 'current_away_scored'] = current_scored
                        current_scored += team_season_data.loc[idx, 'away_scored']

                        data_with_current_points.loc[idx, 'current_away_missed'] = current_missed
                        current_missed += team_season_data.loc[idx, 'home_scored']

                        data_with_current_points.loc[idx, 'current_away_diff'] = current_scored - current_missed

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

        result = self.raw_data.merge(data_with_current_points, how='left')

        return result

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

    def log_features(self) -> NoReturn:

        for feature in self.money_features:
            self.raw_data[f"log_{feature}"] = self.raw_data[feature].apply(np.log)

