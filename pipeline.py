import yaml
from datetime import datetime

import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

from data_transformer import DataTransformer
from scorer import ROIChecker

ENGLAND_DATA_PATH = 'data/england.csv.gz'
FRANCE_DATA_PATH = 'data/france.csv.gz'
GERMANY_DATA_PATH = 'data/germany.csv.gz'
ITALY_DATA_PATH = 'data/italy.csv.gz'
SPAIN_DATA_PATH = 'data/spain.csv.gz'
FEATURES_PATH = 'data/features.yaml'

raw_england_data = pd.read_csv(ENGLAND_DATA_PATH, compression='gzip')
raw_france_data = pd.read_csv(FRANCE_DATA_PATH, compression='gzip')
raw_germany_data = pd.read_csv(GERMANY_DATA_PATH, compression='gzip')
raw_italy_data = pd.read_csv(ITALY_DATA_PATH, compression='gzip')
raw_spain_data = pd.read_csv(SPAIN_DATA_PATH, compression='gzip')

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

cat_features = list(categorical_features)


def get_cv_data_by_time(
    data,
    start_from_season: str = '2017-2018',
    unit: str = 'weeks',
    days: int = 7,
    weeks: int = 1,
    benchmark_league: str = 'premier-league',
    start_date=None,
    finish_date=None,
):
    if start_date:
        start_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    else:
        start_time = int(data[(data.season == start_from_season) & (data.league == benchmark_league)].timestamp_date.min())
    if finish_date:
        finish_time = int(datetime.strptime(finish_date, '%Y-%m-%d').timestamp())
    else:
        finish_time = int(data.timestamp_date.max())

    window = 24 * 3600
    if unit == 'days':
        window *= days
    elif unit == 'weeks':
        window *= weeks * 7
    print(start_time)
    print(finish_time)
    print(window)
    for period in range(start_time, finish_time, window):
        train_cv = data.loc[data.timestamp_date < period]
        val_cv = data.loc[(data.timestamp_date > period) & (data.timestamp_date <= period + window)]
        yield train_cv, val_cv


cv_scorer = ROIChecker()

model_params = {
    'loss_function': 'Logloss',
    'depth': 1, 'iterations': 2500, 'colsample_bylevel': 0.06, 'boosting_type': 'Plain',
    'bootstrap_type': 'Bayesian', 'learning_rate': 0.05, 'l2_leaf_reg': 20, 'auto_class_weights': 'None',
    'random_strength': 7, 'bagging_temperature': 5,
    'verbose': 1000,
    'random_state': 322,
}

both_model_params = {
    'loss_function': 'Logloss',
    'depth': 1, 'iterations': 2500, 'colsample_bylevel': 0.06, 'boosting_type': 'Plain',
    'bootstrap_type': 'Bayesian', 'learning_rate': 0.05, 'l2_leaf_reg': 20, 'auto_class_weights': 'None',
    'random_strength': 7, 'bagging_temperature': 5,
    'verbose': 1000,
    'random_state': 322,
}

cv_model_result = CatBoostClassifier(**{
    'loss_function': 'MultiClass',
    'depth': 6, 'iterations': 2500, 'colsample_bylevel': 0.06, 'boosting_type': 'Plain',
    'bootstrap_type': 'Bayesian', 'learning_rate': 0.05, 'l2_leaf_reg': 20, 'auto_class_weights': 'None',
    'random_strength': 7, 'bagging_temperature': 5,
    'verbose': 1000,
    'random_state': 322,
})
cv_model_total = CatBoostClassifier(**model_params)
cv_model_both = CatBoostClassifier(**both_model_params)

iteration = 0

print(list(cat_features))

START_DATE = '2021-07-01'
FINISH_DATE = '2021-06-30'
for cv_train, cv_test in get_cv_data_by_time(train, start_date=START_DATE):
    if cv_test.shape[0] == 0:
        print(f"Iteration #{iteration} skipped, no matches")
        print('_______')
        iteration += 1
        continue
    cv_y_train_result = cv_train.result_target
    cv_y_train_total = cv_train.total_target
    cv_y_train_both = cv_train.both_target
    cv_X_train = cv_train.drop(columns=['result_target', 'total_target', 'both_target'])

    cv_model_result.fit(cv_X_train, cv_y_train_result, cat_features=cat_features)
    # cv_model_total.fit(cv_X_train, cv_y_train_total, cat_features=cat_features)
    # cv_model_both.fit(cv_X_train, cv_y_train_both, cat_features=cat_features)

    result_target = cv_test.result_target
    total_target = cv_test.total_target
    both_target = cv_test.both_target
    cv_X_test = cv_test.drop(columns=['result_target', 'total_target', 'both_target'])

    preds_proba_result = cv_model_result.predict_proba(cv_X_test)
    # preds_proba_total = cv_model_total.predict_proba(cv_X_test)
    # preds_proba_both = cv_model_both.predict_proba(cv_X_test)

    imp = dict(zip(cv_X_test.columns, cv_model_result.get_feature_importance()))
    print(sorted(imp.items(), key=lambda item: item[1], reverse=True))

    cv_scorer.run_check(
        cv_X_test,
        result_target,
        total_target,
        both_target,
        preds_proba_result=preds_proba_result,
        # preds_proba_total=preds_proba_total,
        # preds_proba_both=preds_proba_both,
    )

    print(f"Iteration #{iteration} complete!")
    print('_______')
    iteration += 1

dynamic_info, static_info = cv_scorer.return_info()
sns.lineplot(x=[i for i in range(len(static_info[2]))], y=static_info[2])
plt.savefig("low_features.png")
