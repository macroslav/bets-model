import pandas as pd
from catboost import CatBoostClassifier, Pool

from typing import List


class Optimizer:

    def __init__(self, train: pd.DataFrame, val: pd.DataFrame, target: str, cat_features: List[str]):
        """
         Initialize Optuna optimizer

        """
        self.train_pool = Pool(data=train.drop(columns=[target]),
                               label=train[target],
                               cat_features=cat_features
                               )

        self.valid_pool = Pool(data=val.drop(columns=[target]),
                               label=val[target],
                               cat_features=cat_features
                               )

    def objective(self, trial):

        param = {
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "3gb",
        }

        gbm = CatBoostClassifier(**param)

        gbm.fit(self.train_pool, eval_set=self.valid_pool, verbose=0, early_stopping_rounds=100)

        preds = gbm.predict(valid_x)
        pred_labels = np.rint(preds)
        accuracy = accuracy_score(valid_y, pred_labels)

        return accuracy
