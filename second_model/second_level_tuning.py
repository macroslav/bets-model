import pandas as pd
from catboost import CatBoostClassifier
from datetime import datetime

import optuna

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


categorical_features = tuple(data.select_dtypes(include=['object']).columns)
print(categorical_features)


def objective(trial):
    params = {
        'depth': trial.suggest_int('depth', 1, 10),
        'iterations': trial.suggest_int('iterations', 50, 10000),
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
    }
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1)
    iteration = 0
    results = []
    START_DATE = '2022-01-01'
    FINISH_DATE = '2022-03-01'
    model = CatBoostClassifier(**params)
    sum_roi = 0.0
    sum_win_rate = 0.0
    sum_coef = 0.0
    sum_matches = 0
    sum_wins = 0
    for cv_train, cv_test in get_cv_data_by_time(data, start_date=START_DATE, finish_date=FINISH_DATE):
        if cv_test.shape[0] == 0:
            print(f"Iteration #{iteration} skipped, no matches")
            print('_______')
            iteration += 1
            continue
        train_target = cv_train['is_correct'].values
        cv_X_train = cv_train.drop('is_correct', axis=1)
        model.fit(cv_X_train, train_target, cat_features=categorical_features)

        test_target = cv_test['is_correct'].values
        cv_X_test = cv_test.drop('is_correct', axis=1)
        predictions = model.predict_proba(cv_X_test)

        imp = dict(zip(cv_X_test.columns, model.get_feature_importance()))
        print(sorted(imp.items(), key=lambda item: item[1], reverse=True))

        matches = 0
        coefs = []
        wins = 0
        x_test = cv_test.reset_index(drop=True)
        for i, elem in enumerate(predictions):
            if elem[1] > 0.5:
                matches += 1
                if 1 == test_target[i]:
                    wins += 1
                    coefs.append(x_test.loc[i]['coef'])

        win_rate = wins / matches if matches != 0 else 0.0
        coef = sum(coefs) / wins if wins != 0 else 0.0
        roi = (win_rate * coef - 1) * 100 if coef != 0.0 else 0.0
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
        sum_roi = (sum_win_rate * sum_coef - 1) * 100 if sum_coefs != 0.0 else 0.0

        print({'roi': sum_roi, 'win_rate': sum_win_rate, 'coef': sum_coef, 'matches': sum_matches, 'wins': sum_wins})

        iteration += 1
        print(f"Iteration #{iteration} complete!")
        print('_______')
    with open('second_tuning.txt', 'a') as f:
        f.write(str(params) + '\n')
        f.write(str({'roi': sum_roi, 'win_rate': sum_win_rate, 'coef': sum_coef, 'matches': sum_matches, 'wins': sum_wins}) + '\n')
        f.write('_____' + '\n')
    return sum_roi


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)
print(study.best_trial)
print(study.best_trial.value)
for key, value in study.best_trial.params.items():
    print(key, value)


