import yaml

import optuna
import pandas as pd
from catboost import CatBoostClassifier

from data_transformer import DataTransformer


DATA_PATH = 'data/full_top_5_leagues.csv'
FEATURES_PATH = 'data/features.yaml'

raw_train_data = pd.read_csv(DATA_PATH)

raw_train_data = raw_train_data.sort_values(by='season')
raw_train_data.reset_index(inplace=True, drop=True)

with open(FEATURES_PATH) as f:
    all_features_dict = yaml.safe_load(f)

for key, item in all_features_dict.items():
    if isinstance(item, dict):
        print(f"'{key}':")
        for inner_key in item.keys():
            print(f"\t'{inner_key}'")
    else:
        print(f"'{key}'")


def base_data_preprocess(data):
    preprocessed_data = data.copy()
    preprocessed_data = preprocessed_data.sort_values(by='timestamp_date')
    preprocessed_data = preprocessed_data.drop(columns=['date', 'link'])
    drop_index = preprocessed_data[preprocessed_data.home_goalkeepers_average_age.isna()].index
    preprocessed_data = preprocessed_data.drop(index=drop_index)

    return preprocessed_data


train_data = base_data_preprocess(raw_train_data)

numeric_features = tuple(train_data.select_dtypes(include=['int', 'float']).columns)

categorical_features = tuple(train_data.select_dtypes(include=['object']).columns)

features = {'cat_features': categorical_features,
            'num_features': numeric_features,
            'grouped_features': all_features_dict
            }

transformer_context = {'data': train_data,
                       'features': features
                       }

transformer = DataTransformer(transformer_context)
train, decode_labels = transformer.run_logic()
train = train.reset_index(drop=True)

train_dataset = train.loc[:40500]
test_dataset = train.loc[40500:].reset_index()


y_train = train_dataset.result_target.reset_index()['result_target']
y_test = test_dataset.result_target.reset_index()['result_target']
train_dataset = train_dataset.drop(columns=['result_target', 'total_target', 'both_target'])
test_dataset = test_dataset.drop(columns=['result_target', 'total_target', 'both_target'])


def make_predictions(proba, test):
    results = []
    for i, row in test.iterrows():
        home_value = proba[i][2] * row['home_win_rate']
        away_value = proba[i][0] * row['away_win_rate']
        draw_value = proba[i][1] * row['draw_rate']
        values = [home_value, away_value, draw_value]
        for index, value in enumerate(values):
            if value > 1:
                if index == 0:
                    bet = 3
                    coef = row['home_win_rate']
                    chance = proba[i][2]
                elif index == 1:
                    bet = 0
                    coef = row['away_win_rate']
                    chance = proba[i][0]
                else:
                    bet = 1
                    coef = row['draw_rate']
                    chance = proba[i][1]
                result = {
                    'league': row['league'],
                    'season': row['season'],
                    'bet': bet,
                    'coef': coef,
                    'chance': chance,
                    'date': row['timestamp_date'],
                    'index': i,
                }
                results.append(result)
    return results


def get_score(predictions, target):

    total_profit = 0

    for index, row in enumerate(predictions):
        accepted_coef = row['coef']
        i = row['index']

        if target[i] == row['bet']:
            total_profit += 100 * (accepted_coef - 1)
        else:
            total_profit -= 100
    print(total_profit)
    return total_profit


def objective(trial):
    params = {
        'depth': 1,
        'iterations': trial.suggest_int('iterations', 5000, 7000),
        'colsample_bylevel': 0.09805089887364203,
        "boosting_type": "Ordered",
        "bootstrap_type": "Bayesian",
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 5, 20),
        'random_state': 322,
        'verbose': 250,
        'loss_function': 'MultiClass',
        'bagging_temperature': 4.087227192055693,
        'used_ram_limit': '2gb',
    }
    model = CatBoostClassifier(**params)
    model.fit(train_dataset, y_train)
    proba = model.predict_proba(test_dataset)
    predictions = make_predictions(proba, test_dataset)
    return get_score(predictions, y_test)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000)
print(study.best_trial)
print(study.best_trial.value)
for key, value in study.best_trial.params.items():
    print(key, value)
