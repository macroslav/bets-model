import numpy as np
from catboost import CatBoostClassifier, Pool

from typing import Protocol, Dict, Union
from pathlib import Path
from datetime import datetime

from configs.paths import MODELS_DIR


class BaseModel(Protocol):
    """
        Implementation of base model class
    """

    def fit(self):
        """ Fit model """

    def predict(self):
        """ Get predicts for test data"""

    def get_feature_importances(self):
        ...


class ClassifierBoostingModel:
    """
        Implementation of boosting model
    """

    def __init__(self, params: Dict, data: Dict):

        self.params = params

        self.model = CatBoostClassifier(**self.params)

        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_path = MODELS_DIR / Path(f'catboost_{current_datetime}')

    def fit(self, train_data: Pool, valid_data: Pool):
        self.model.fit(train_data, eval_set=valid_data)

    def predict(self, test_data) -> np.ndarray:

        return self.model.predict(test_data)

    def predict_proba(self, test_data) -> np.ndarray:

        return self.model.predict_proba(test_data)

    def get_feature_importances(self, importance_type: str = 'catboost'):

        if importance_type == 'catboost':
            return self.model.get_feature_importance()

        elif importance_type == 'permutation':
            return

    def save_model(self) -> None:
        self.model.save_model(self.save_path)
