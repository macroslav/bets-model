import pandas as pd
from catboost import CatBoostClassifier
from datetime import datetime

data = pd.read_csv('file.csv')
data['country_league'] = data['country'] + data['league']
print(data.shape)


def get_cv_data_by_time(
    data,
    start_date: str,
    unit: str = 'weeks',
    days: int = 7,
    weeks: int = 1,
    finish_date=None,
):
    start_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    if finish_date:
        finish_time = int(datetime.strptime(finish_date, '%Y-%m-%d').timestamp())
    else:
        finish_time = int(data.date.max())

    window = 24 * 3600
    if unit == 'days':
        window *= days
    elif unit == 'weeks':
        window *= weeks * 7
    print(start_time)
    print(finish_time)
    print(window)
    for period in range(start_time, finish_time, window):
        train_cv = data.loc[data.date < period]
        val_cv = data.loc[(data.date > period) & (data.date <= period + window)]
        yield train_cv, val_cv


model_params = {
    'depth': 9, 'iterations': 6001, 'colsample_bylevel': 0.0756403345124913, 'boosting_type': 'Ordered',
    'bootstrap_type': 'Bayesian', 'learning_rate': 0.2561863099356313, 'l2_leaf_reg': 12, 'auto_class_weights': 'None',
    'random_strength': 3, 'random_state': 322, 'verbose': 1000, 'loss_function': 'Logloss',
    'bagging_temperature': 3.8582336859155353
}
model = CatBoostClassifier(**model_params)

categorical_features = tuple(data.select_dtypes(include=['object']).columns)
print(categorical_features)

iteration = 0
results = []
default_results = []
START_DATE = '2021-07-01'
FINISH_DATE = '2021-07-01'
for cv_train, cv_test in get_cv_data_by_time(data, start_date=START_DATE):
    if cv_test.shape[0] == 0:
        print(f"Iteration #{iteration} skipped, no matches")
        print('_______')
        iteration += 1
        continue
    train_target = cv_train['is_correct'].values
    cv_X_train = cv_train.drop('is_correct', axis=1)
    # cv_X_train = cv_X_train[['chance', 'value', 'bet']]
    model.fit(cv_X_train, train_target, cat_features=categorical_features)

    test_target = cv_test['is_correct'].values
    cv_X_test = cv_test.drop('is_correct', axis=1)
    # cv_X_test = cv_X_test[['chance', 'value', 'bet']]
    predictions = model.predict_proba(cv_X_test)

    imp = dict(zip(cv_X_test.columns, model.get_feature_importance()))
    print(sorted(imp.items(), key=lambda item: item[1], reverse=True))

    matches = 0
    coefs = []
    wins = 0
    x_test = cv_test.reset_index(drop=True)
    for i, elem in enumerate(predictions):
        if elem[1] > 0.5:
            # print(x_test.loc[i].to_dict())
            matches += 1
            if 1 == test_target[i]:
                wins += 1
                coefs.append(x_test.loc[i]['coef'])

    win_rate = wins / matches if matches != 0 else 0.0
    coef = sum(coefs) / wins if wins != 0 else 0.0
    roi = (win_rate * coef - 1) * 100
    results.append({'matches': matches, 'wins': wins, 'win_rate': win_rate, 'coef': coef, 'roi': roi})

    sum_matches = 0
    sum_wins = 0
    sum_coefs = 0
    for result in results:
        sum_matches += result['matches']
        sum_wins += result['wins']
        sum_coefs += result['wins'] * result['coef']
    sum_win_rate = sum_wins / sum_matches if sum_matches > 0 else 0.0
    sum_coef = sum_coefs / sum_wins if sum_wins > 0 else 0.0
    sum_roi = (sum_win_rate * sum_coef - 1) * 100

    print({'roi': sum_roi, 'win_rate': sum_win_rate, 'coef': sum_coef, 'matches': sum_matches, 'wins': sum_wins})
    print('_______')

    default_matches = 0
    default_coefs = []
    default_wins = 0
    x_test = cv_test.reset_index(drop=True)
    for i, elem in enumerate(test_target):
        if x_test.loc[i]['value'] > 1:
            default_matches += 1
            if 1 == test_target[i]:
                default_wins += 1
                default_coefs.append(x_test.loc[i]['coef'])

    default_win_rate = default_wins / default_matches if default_matches != 0 else 0.0
    default_coef = sum(default_coefs) / default_wins if default_wins != 0 else 0.0
    default_roi = (default_win_rate * default_coef - 1) * 100
    default_results.append({'matches': default_matches, 'wins': default_wins, 'win_rate': default_win_rate, 'coef': default_coef, 'roi': default_roi})

    default_sum_matches = 0
    default_sum_wins = 0
    default_sum_coefs = 0
    for result in default_results:
        default_sum_matches += result['matches']
        default_sum_wins += result['wins']
        default_sum_coefs += result['wins'] * result['coef']
    default_sum_win_rate = default_sum_wins / default_sum_matches if default_sum_matches > 0 else 0.0
    default_sum_coef = default_sum_coefs / default_sum_wins if default_sum_wins > 0 else 0.0
    default_sum_roi = (default_sum_win_rate * default_sum_coef - 1) * 100

    iteration += 1
    print({'default_roi': default_sum_roi, 'default_win_rate': default_sum_win_rate, 'default_coef': default_sum_coef, 'default_matches': default_sum_matches, 'default_wins': default_sum_wins})
    print(f"Iteration #{iteration} complete!")
    print('_______')
