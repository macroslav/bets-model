from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from typing import Protocol, Dict, List, Optional, Any

from config import ModelConfig


class BaseModel(Protocol):

    """
        Implementation of base model class
    """

    def __init__(self):
        ...

    def fit(self):
        ...

    def predict(self):
        ...

    def predict_proba(self):
        ...

    def get_feature_importances(self):
        ...


class BoostingModel:
    """
        Implementation of boosting model
    """

    def __init__(self, params: Dict, data: Dict):

        self.params = params
        self.train = data['train']
        self.val = data['val']
        self.test = data['test']
        self.target = data['target']
        self.cat_features = data['cat_features']

        self.model = CatBoostClassifier(**self.params)

        self.train_pool = None
        self.valid_pool = None
        self.test_pool = None

        self.create_pool()

        self.preds_proba = None
        self.preds_class = None

    def create_pool(self):

        self.train_pool = Pool(data=self.train.drop(columns=[self.target]),
                               label=self.train[self.target],
                               cat_features=self.cat_features
                               )

        self.valid_pool = Pool(data=self.val.drop(columns=[self.target]),
                               label=self.val[self.target],
                               cat_features=self.cat_features
                               )

        self.test_pool = Pool(data=self.test.drop(columns=[self.target]),
                              label=self.test[self.target],
                              cat_features=self.cat_features
                              )

    def fit(self):
        self.model.fit(self.train_pool, eval_set=self.valid_pool)

    def predict(self, test_data):

        self.preds_class = None

        self.preds_class = self.model.predict(test_data)

        return self.preds_class

    def predict_proba(self, test_data):
        if self.preds_proba:
            self.preds_proba = None

        self.preds_proba = self.model.predict_proba(test_data)

        return self.preds_proba

    def get_feature_importances(self):
        return self.model.get_feature_importance()
