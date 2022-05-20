from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# import tensorflow as tf

from typing import Protocol, Dict, List, Optional, Any

from config import ModelConfig


class BaseModel(Protocol):
    """
        Implementation of base model class
    """

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

        self.preds_proba = None

        self.preds_proba = self.model.predict_proba(test_data)

        return self.preds_proba

    def get_feature_importances(self, importance_type='catboost'):

        """
        Return feature importances depends on selected importance_type: catboost, permutation or shap

        Parameters
        ----------

        importance_type : str, default = 'catboost'

        Allowed values are :

        - 'catboost', return catboost built-in values of feature importance (CatBoostModel.get_feature_importance())
        - 'permutation', return permutation importance values with eli5
        - 'shap', return feature importance based on Shapley's vectors

        """

        if importance_type == 'catboost':
            return self.model.get_feature_importance()

        elif importance_type == 'permutation':
            return


class RegressionNeuralNetwork:

    def __init__(self, data):
        self.train = data['train']
        self.val = data['val']
        self.test = data['test']
        self.target = data['target']
        self.cat_features = data['cat_features']

        self.input_shape = self.train.shape

        self.model = tf.keras.Sequential()

    def fit(self):
        self.model.fit()

    def build_and_compile_model(self, norm):
        self.model = tf.keras.Sequential([
            norm,
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(loss='mean_absolute_error',
                           optimizer=tf.keras.optimizers.Adam(0.001))
