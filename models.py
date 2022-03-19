from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier


class BaseModel:
    def __init__(self):
        pass

