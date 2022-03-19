import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, NoReturn


class DataTransformer:

    def __init__(self, context: Dict):
        self.train_data = context['train']
        self.test_data = context['test']
        self.val_data = pd.DataFrame()
        self.cat_features = context['cat_features']
        self.num_features = context['num_features']

    def run_logic(self):
        self._fill_nans()
        self._set_target()
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

    def _set_target(self) -> NoReturn:
        pass

    def _team_encoding(self) -> NoReturn:
        pass

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
