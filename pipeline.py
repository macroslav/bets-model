import yaml

import pandas as pd
import numpy as np
import seaborn as sns
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

from data_transformer import DataTransformer
from scorer import ROIChecker

DATA_PATH = 'data/top_5_leagues.csv'
FEATURES_PATH = 'data/features.yaml'

raw_train_data = pd.read_csv(DATA_PATH)


# raw_train_data = raw_train_data[raw_train_data['country'] == 'france']

raw_train_data = raw_train_data.sort_values(by='season')
raw_train_data.reset_index(inplace=True, drop=True)


"""
for i, row in raw_train_data.iterrows():
    if row['season'] == '2021-2022':
        print(i)
"""


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

    preprocessed_data['day'] = pd.to_datetime(preprocessed_data.date, format='%d.%m.%Y')
    preprocessed_data['year'] = preprocessed_data['day'].dt.year
    preprocessed_data.date = preprocessed_data.day.values.astype(np.int64) // 10 ** 9
    preprocessed_data = preprocessed_data.sort_values(by='date')
    preprocessed_data = preprocessed_data.drop(columns=['link'])
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
train, val, test, decode_labels, encode_labels = transformer.run_logic()

cat_features = list(categorical_features)


def get_cv_data(train_data, train_initial_size=36050, window=5000):
    for index in range(0, train_data.shape[0] - train_initial_size - window + 1, window):
        train_cv = train_data.loc[:train_initial_size + index]
        val_cv = train_data.loc[train_initial_size + index: train_initial_size + index + window]
        yield train_cv, val_cv


cv_scorer = ROIChecker(decode_labels=decode_labels['team_names'])

model_params = {'n_estimators': 1000,
                'learning_rate': 0.03,
                'loss_function': 'MultiClass',
                'verbose': 250
                }

cv_model = CatBoostClassifier(**model_params)

iteration = 0
for cv_train, cv_test in get_cv_data(train):
    cv_y_train = cv_train.target
    cv_X_train = cv_train.drop(columns=['target'])
    cv_y_test = cv_test.target
    cv_X_test = cv_test.drop(columns=['target'])

    cv_model.fit(cv_X_train, cv_y_train)

    preds_proba = cv_model.predict_proba(cv_X_test)
    preds_class = cv_model.predict(cv_X_test)

    cv_scorer.run_check(cv_X_test, cv_y_test, preds_proba, preds_class)

    iteration += 1
    print(f"Iteration #{iteration} complete!")
    print('_______')

dynamic_info, static_info = cv_scorer.return_info()
from pprint import pprint
pprint(static_info)
sns.lineplot(x=[i for i in range(len(static_info[2]))], y=static_info[2])
plt.savefig("top_5_leagues_2021.png")
