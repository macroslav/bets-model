from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from typing import Protocol


class BaseModel(Protocol):

    """
        Implementation of base model class
    """

    def __init__(self):
        pass


class BoostingModel:
    """
        Implementation of boosting model
    """