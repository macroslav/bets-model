import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
        self.val_data = pd.DataFrame()
        self.cat_features = context['cat_features']
        self.num_features = context['num_features']

    def run_logic(self):
        self._fill_nans()
        self.train_data['target'] = self.train_data.apply(_set_target, axis=1)
        self.test_data['target'] = self.test_data.apply(_set_target, axis=1)
        self._team_encoding()
        self._categorical_encoding()
        self._generate_features()
        self._split()
        return self.train_data, self.val_data, self.test_data

    def _fill_nans(self) -> NoReturn:
        pass

    def _categorical_encoding(self) -> NoReturn:
        pass

    def _split(self) -> NoReturn:
        pass

    def _team_encoding(self) -> NoReturn:

        all_teams = self.train_data.sort_values(by='home_team').home_team.unique()
        self.teams_labels = {team: number for number, team in enumerate(all_teams)}
        self.decode_teams = {number: team for number, team in enumerate(all_teams)}

        self.train_data['home_team'] = self.train_data['home_team'].map(self.teams_labels)
        self.train_data['away_team'] = self.train_data['away_team'].map(self.teams_labels)
        self.test_data['home_team'] = self.test_data['home_team'].map(self.teams_labels)
        self.test_data['away_team'] = self.test_data['away_team'].map(self.teams_labels)

    def _generate_features(self) -> NoReturn:
        self._season_averages()
        self._current_position()
        self._alltime_averages()
        self._win_lose_streak()
        self._season_totals()

    def _season_averages(self) -> NoReturn:
        pass

    def _season_totals(self) -> NoReturn:
        pass

    def _current_position(self) -> NoReturn:
        pass

    def _win_lose_streak(self) -> NoReturn:
        pass

    def _alltime_averages(self) -> NoReturn:
        pass

    def _max_results(self) -> NoReturn:
        pass
