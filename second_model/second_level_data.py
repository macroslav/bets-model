import ast
import pandas as pd
from tqdm import tqdm

with open('file.txt', 'r') as f:
    data = f.read()

new_data = []
for elem in data.split('\n'):
    try:
        new_data.append(ast.literal_eval(elem))
    except Exception:
        print(elem)
        pass

data = pd.DataFrame(new_data)


ENGLAND_DATA_PATH = 'data/england.csv.gz'
FRANCE_DATA_PATH = 'data/france.csv.gz'
GERMANY_DATA_PATH = 'data/germany.csv.gz'
ITALY_DATA_PATH = 'data/italy.csv.gz'
SPAIN_DATA_PATH = 'data/spain.csv.gz'
RUSSIA_DATA_PATH = 'data/russia.csv.gz'
PORTUGAL_DATA_PATH = 'data/portugal.csv.gz'
FEATURES_PATH = 'data/features.yaml'

raw_england_data = pd.read_csv(ENGLAND_DATA_PATH, compression='gzip')
raw_france_data = pd.read_csv(FRANCE_DATA_PATH, compression='gzip')
raw_germany_data = pd.read_csv(GERMANY_DATA_PATH, compression='gzip')
raw_italy_data = pd.read_csv(ITALY_DATA_PATH, compression='gzip')
raw_spain_data = pd.read_csv(SPAIN_DATA_PATH, compression='gzip')
raw_russia_data = pd.read_csv(RUSSIA_DATA_PATH, compression='gzip')
raw_portugal_data = pd.read_csv(PORTUGAL_DATA_PATH, compression='gzip')

raw_train_data = pd.concat(
    [
        raw_england_data,
        raw_france_data,
        raw_germany_data,
        raw_italy_data,
        raw_spain_data,
        raw_russia_data,
        raw_portugal_data,
    ],
    ignore_index=True,
)

# data = data[data['value'] > 1].reset_index(drop=True)

league_types = []
home_league_current_season_win_rate_at_home = []
away_league_current_season_win_rate_at_away = []
home_manager_working_days = []
away_manager_working_days = []
home_real_total_market_value = []
away_real_total_market_value = []
months = []
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    match_data = raw_train_data.loc[
        (raw_train_data['home_team'] == row['home_team']) &
        (raw_train_data['timestamp_date'] == row['date'])
        ].reset_index(drop=True)
    row_match_data = match_data.loc[0]
    league_types.append(row_match_data['league_type'])
    # home_league_current_season_win_rate_at_home.append(row_match_data['home_league_current_season_win_rate_at_home'])
    # away_league_current_season_win_rate_at_away.append(row_match_data['away_league_current_season_win_rate_at_away'])
    # home_manager_working_days.append(row_match_data['home_manager_working_days'])
    # away_manager_working_days.append(row_match_data['away_manager_working_days'])
    # months.append(row_match_data['month'])
    # home_real_total_market_value.append(row_match_data['home_real_total_market_value'])
    # away_real_total_market_value.append(row_match_data['away_real_total_market_value'])

data['league_type'] = league_types
# data['home_league_current_season_win_rate_at_home'] = home_league_current_season_win_rate_at_home
# data['away_league_current_season_win_rate_at_away'] = away_league_current_season_win_rate_at_away
# data['home_manager_working_days'] = home_manager_working_days
# data['away_manager_working_days'] = away_manager_working_days
# data['month'] = months
# data['home_real_total_market_value'] = home_real_total_market_value
# data['away_real_total_market_value'] = away_real_total_market_value
data.to_csv('file.csv', index=False)
