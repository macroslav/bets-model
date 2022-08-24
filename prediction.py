import json
import yaml
from contextlib import closing
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
from catboost import CatBoostClassifier

from data_transformer import DataTransformer

ENGLAND_DATA_PATH = 'data/england.csv'
FRANCE_DATA_PATH = 'data/france.csv'
GERMANY_DATA_PATH = 'data/germany.csv'
ITALY_DATA_PATH = 'data/italy.csv'
SPAIN_DATA_PATH = 'data/spain.csv'
FEATURES_PATH = 'data/features.yaml'


class Predictor:

    def __init__(self, dataset):

        raw_england_data = pd.read_csv(ENGLAND_DATA_PATH)
        raw_france_data = pd.read_csv(FRANCE_DATA_PATH)
        raw_germany_data = pd.read_csv(GERMANY_DATA_PATH)
        raw_italy_data = pd.read_csv(ITALY_DATA_PATH)
        raw_spain_data = pd.read_csv(SPAIN_DATA_PATH)

        self.raw_train_data = pd.concat(
            [
                raw_england_data,
                raw_france_data,
                raw_germany_data,
                raw_italy_data,
                raw_spain_data,
            ],
            ignore_index=True,
        )
        self.raw_train_data.reset_index(inplace=True, drop=True)

        self.raw_future_data = dataset
        self.raw_future_data.reset_index(inplace=True, drop=True)

        self.future_data = None
        self.links = None

        with open(FEATURES_PATH) as f:
            self.all_features_dict = yaml.safe_load(f)

    def run(self, user_data: dict):
        preds_proba = self.predict()
        print('Predict done')
        results = self.make_results(preds_proba)
        print('Results done')
        self.save_to_json(results)
        print('Saved to json')
        self.save_to_db(user_data, results)
        print('Saved to db')

    def base_data_preprocess(self, data):
        preprocessed_data = data.copy()
        preprocessed_data = preprocessed_data.sort_values(by='timestamp_date')
        self.links = preprocessed_data['link'].values
        preprocessed_data = preprocessed_data.drop(columns=['date', 'link'])
        drop_index = preprocessed_data[preprocessed_data.home_goalkeepers_average_age.isna()].index
        preprocessed_data = preprocessed_data.drop(index=drop_index)

        return preprocessed_data

    def predict(self):
        train_data = self.base_data_preprocess(self.raw_train_data)
        self.future_data = self.base_data_preprocess(self.raw_future_data)

        numeric_features = tuple(train_data.select_dtypes(include=['int', 'float']).columns)
        future_numeric_features = tuple(self.future_data.select_dtypes(include=['int', 'float']).columns)

        categorical_features = tuple(train_data.select_dtypes(include=['object']).columns)
        future_categorical_features = tuple(self.future_data.select_dtypes(include=['object']).columns)

        features = {
            'cat_features': categorical_features,
            'num_features': numeric_features,
            'grouped_features': self.all_features_dict,
        }

        future_features = {
            'cat_features': future_categorical_features,
            'num_features': future_numeric_features,
            'grouped_features': self.all_features_dict,
        }

        transformer_context = {
            'data': train_data,
            'features': features,
        }
        future_transformer_context = {
            'data': self.future_data,
            'features': future_features,
        }

        transformer = DataTransformer(transformer_context)
        train, _ = transformer.run_logic()
        print('finish transformer')

        model_params = {
            'loss_function': 'MultiClass',
            'depth': 1,
            'iterations': 6884,
            'colsample_bylevel': 0.09805089887364203,
            'boosting_type': 'Ordered',
            'bootstrap_type': 'Bayesian',
            'learning_rate': 0.0895929273549625,
            'l2_leaf_reg': 12,
            'bagging_temperature': 4.087227192055693,
            'verbose': 250,
            'random_state': 322,
        }

        model = CatBoostClassifier(**model_params)
        y_train = train.result_target
        X_train = train.drop(columns=['result_target', 'total_target', 'both_target'])
        model.fit(X_train, y_train)

        future_transformer = DataTransformer(future_transformer_context)
        future_train = future_transformer.run_future_logic()
        print('finish  future transformer')
        return model.predict_proba(future_train)

    def make_results(self, preds_proba: list[list[float]]) -> list[dict]:
        results = []
        for i, row in self.future_data.iterrows():
            home_value = preds_proba[i][2] * row['home_win_rate']
            away_value = preds_proba[i][0] * row['away_win_rate']
            draw_value = preds_proba[i][1] * row['draw_rate']
            values = [home_value, away_value, draw_value]
            max_value = values.index(max(values))
            if values[max_value] > 1:
                print(f"{row['home_team']} - {row['away_team']}  {row['country']}, {row['league']}\n")
                if max_value == 0:
                    bet = '1'
                    coef = row['home_win_rate']
                    chance = preds_proba[i][2]
                elif max_value == 1:
                    bet = '2'
                    coef = row['away_win_rate']
                    chance = preds_proba[i][0]
                else:
                    bet = '0'
                    coef = row['draw_rate']
                    chance = preds_proba[i][1]
                current_datetime = datetime.fromtimestamp(row['timestamp_date']) + timedelta(hours=3)
                result = {
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'league': row['league'],
                    'country': row['country'],
                    'season': row['season'],
                    'league_level': str(row['league_level']),
                    'timestamp_date': row['timestamp_date'],
                    'bet': bet,
                    'bet_type': 'result',
                    'coef': coef,
                    'chance': chance,
                    'day': current_datetime.strftime('%d.%m.%Y'),
                    'hours': current_datetime.strftime('%H:%M'),
                    'link': self.links[i],
                }
                results.append(result)
        return results

    def save_to_db(self, user_data: dict, results: list[dict]):
        with closing(psycopg2.connect(**user_data)) as conn:
            with conn.cursor() as cursor:
                for result in results:
                    cursor.execute(
                        "INSERT INTO predictions"
                        "(home_team, away_team, league, country, season, league_level, timestamp_date, bet, bet_type, coef, chance, day, hours, link)"
                        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        list(result.values()),
                    )
                    conn.commit()

    def save_to_json(self, results: list[dict]):
        with open('results.json', 'w') as f:
            json.dump(results, f, ensure_ascii=False)
