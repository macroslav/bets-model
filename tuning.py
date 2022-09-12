import yaml

import optuna
import pandas as pd
from catboost import CatBoostClassifier

from data_transformer import DataTransformer


ENGLAND_DATA_PATH = 'data/england.csv'
FRANCE_DATA_PATH = 'data/france.csv'
GERMANY_DATA_PATH = 'data/germany.csv'
ITALY_DATA_PATH = 'data/italy.csv'
SPAIN_DATA_PATH = 'data/spain.csv'
FEATURES_PATH = 'data/features.yaml'

raw_england_data = pd.read_csv(ENGLAND_DATA_PATH)
raw_france_data = pd.read_csv(FRANCE_DATA_PATH)
raw_germany_data = pd.read_csv(GERMANY_DATA_PATH)
raw_italy_data = pd.read_csv(ITALY_DATA_PATH)
raw_spain_data = pd.read_csv(SPAIN_DATA_PATH)

raw_train_data = pd.concat(
    [
        raw_england_data,
        raw_france_data,
        raw_germany_data,
        raw_italy_data,
        raw_spain_data,
    ],
    ignore_index=True,
)

raw_train_data = raw_train_data.sort_values(by='season')
raw_train_data.reset_index(inplace=True, drop=True)

with open(FEATURES_PATH) as f:
    all_features_dict = yaml.safe_load(f)


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

train_dataset = train.loc[:45000]
test_dataset = train.loc[45000:].reset_index()


y_train = train_dataset.both_target.reset_index()['both_target']
y_test = test_dataset.both_target.reset_index()['both_target']
train_dataset = train_dataset.drop(columns=['result_target', 'total_target', 'both_target'])
test_dataset = test_dataset.drop(columns=['result_target', 'total_target', 'both_target'])

cat_features = list(categorical_features)

print(list(cat_features))


def make_both_predictions(proba, test):
    results = []
    for i, row in test.iterrows():
        yes_value = proba[i][1] * row['both_team_to_score_yes']
        no_value = proba[i][0] * row['both_team_to_score_no']
        values = [no_value, yes_value]
        for index, value in enumerate(values):
            if value > 1:
                if value == 0:
                    bet = 0
                    coef = row['both_team_to_score_no']
                    chance = proba[i][0]
                else:
                    bet = 1
                    coef = row['both_team_to_score_yes']
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


def make_total_predictions(proba, test):
    results = []
    for i, row in test.iterrows():
        over_value = proba[i][1] * row['total_over_25_rate']
        under_value = proba[i][0] * row['total_under_25_rate']
        values = [under_value, over_value]
        for index, value in enumerate(values):
            if value > 1:
                if value == 0:
                    bet = 0
                    coef = row['total_under_25_rate']
                    chance = proba[i][0]
                else:
                    bet = 1
                    coef = row['total_over_25_rate']
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
    #  При получении лучших параметров нужно прогонять на них pipeline.py
    #  И фиксировать результаты в https://trello.com/c/uGpuNYUz
    params = {
        'depth': trial.suggest_int('depth', 1, 5),
        'iterations': trial.suggest_int('iterations', 50, 5000),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.1),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 3, 50),
        'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['None', 'Balanced', 'SqrtBalanced']),
        'random_strength': trial.suggest_int('random_strength', 1, 10),
        'random_state': 322,
        'verbose': 1000,
        'loss_function': 'Logloss',
        'used_ram_limit': '3.5gb',
    }
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1)
    model = CatBoostClassifier(**params)
    model.fit(train_dataset, y_train, cat_features=cat_features)
    proba = model.predict_proba(test_dataset)
    predictions = make_predictions(proba, test_dataset)
    return get_score(predictions, y_test)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)
print(study.best_trial)
print(study.best_trial.value)
for key, value in study.best_trial.params.items():
    print(key, value)
