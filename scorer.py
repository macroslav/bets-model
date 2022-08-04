import numpy as np
import pandas as pd
from typing import Protocol, Dict

RESULTS = {'home': 3, 'draw': 1, 'away': 0}
COLS = {'home': 'home_win_rate', 'draw': 'draw_rate', 'away': 'away_win_rate'}
ROI_KINDS = ['home_win_ROI', 'away_win_ROI', 'draw_ROI']


class BaseScorer(Protocol):
    """

        Implementation of base scorer class

    """


class ROIChecker:

    def __init__(self, decode_labels: Dict,
                 start_bank: int = 10000,
                 static_bet: int = 100,
                 percent_of_bank: float = 0.01,
                 roi_threshold: float = 2):

        self.decoder = decode_labels

        self.static_bet = static_bet
        self.dynamic_bet = start_bank * percent_of_bank
        self.start_bank = start_bank
        self.current_dynamic_bank = self.start_bank
        self.current_static_bank = self.start_bank
        self.percent_of_bank = percent_of_bank
        self.roi_threshold = roi_threshold

        self.detail_static_bank_values = [self.start_bank]
        self.batch_static_bank_values = [self.start_bank]
        self.detail_dynamic_bank_values = [self.start_bank]
        self.batch_dynamic_bank_values = [self.start_bank]
        self.predictions = pd.DataFrame()
        self.full_static_info = []
        self.full_dynamic_info = []

        self.y_test = None
        self.X_test = None
        self.preds_class = None
        self.preds_proba = None

    def run_check(self, X_test: pd.DataFrame, y_test: np.ndarray, preds_proba, preds_class):

        self.preds_proba = preds_proba
        self.preds_class = preds_class
        self.X_test = X_test
        self.y_test = y_test

        self.make_predictions_df()
        roi_info = self.get_roi()
        best_roi_df = self.explain_roi_info(roi_info)

        self.count_static_money(best_roi_df)
        # self.count_dynamic_money(best_roi_df)

    def return_info(self):
        return (self.full_dynamic_info, self.detail_dynamic_bank_values, self.batch_dynamic_bank_values), (
            self.full_static_info, self.detail_static_bank_values, self.batch_static_bank_values)

    def make_predictions_df(self):

        self.predictions = pd.DataFrame()
        self.predictions['home_team'] = self.X_test.home_team.map(self.decoder)
        self.predictions['away_team'] = self.X_test.away_team.map(self.decoder)
        self.predictions['home_win_proba'] = self.preds_proba[:, 0]
        self.predictions['draw_proba'] = self.preds_proba[:, 1]
        self.predictions['away_win_proba'] = self.preds_proba[:, 2]
        self.predictions['home_win_rate'] = self.X_test.home_win_rate
        self.predictions['draw_rate'] = self.X_test.draw_rate
        self.predictions['away_win_rate'] = self.X_test.away_win_rate
        self.predictions['result'] = self.y_test
        self.predictions['predict'] = self.preds_class

    def get_roi(self):

        self.predictions['home_win_ROI'] = self.predictions.home_win_rate * self.predictions.home_win_proba - 1
        self.predictions['away_win_ROI'] = self.predictions.away_win_rate * self.predictions.away_win_proba - 1
        self.predictions['draw_ROI'] = self.predictions.draw_rate * self.predictions.draw_proba - 1

        return self.predictions

    def explain_roi_info(self, predictions_with_roi_info):

        best_roi_df = pd.DataFrame()

        for index, row in predictions_with_roi_info.iterrows():

            max_roi = np.max(predictions_with_roi_info.loc[index, ROI_KINDS])
            current_choice = 'home_win_ROI'

            for col in ROI_KINDS:
                if row[col] == max_roi:
                    current_choice = col

            current_choice = ' '.join(current_choice.split('_')[:1])

            best_roi_df.loc[index, 'home_team'] = row.home_team
            best_roi_df.loc[index, 'away_team'] = row.away_team
            best_roi_df.loc[index, 'best_ROI'] = max_roi
            best_roi_df.loc[index, 'choice'] = current_choice
            best_roi_df.loc[index, 'home_win_rate'] = row.home_win_rate
            best_roi_df.loc[index, 'draw_rate'] = row.draw_rate
            best_roi_df.loc[index, 'away_win_rate'] = row.away_win_rate
            best_roi_df.loc[index, 'result'] = row.result
            best_roi_df.loc[index, 'predict'] = row.predict

        return best_roi_df

    # def count_dynamic_money(self):
    #
    #     self.full_dynamic_info.append(result)
    #     self.dynamic_bank_values.append(total_bank)

    def count_static_money(self, best_roi_df):

        initial_bank = self.current_static_bank

        total_profit = 0
        total_coef = 0

        win_bets = 0
        lose_bets = 0

        skipped_bets = 0
        accepted_bets = 0

        win_coef = 0
        lose_coef = 0

        win_bank = 0
        lose_bank = 0

        for index, row in best_roi_df.iterrows():
            if row.best_ROI > self.roi_threshold:

                accepted_bets += 1
                accepted_coef = row[COLS[row.choice]]

                if RESULTS[row.choice] == row.result:

                    current_profit = self.dynamic_bet * (accepted_coef - 1)
                    total_profit += current_profit
                    self.update_static_bank(self.current_static_bank + current_profit)
                    win_coef += accepted_coef
                    win_bank += current_profit
                    win_bets += 1

                else:
                    total_profit -= self.dynamic_bet
                    self.update_static_bank(self.current_static_bank - self.dynamic_bet)
                    lose_coef += accepted_coef
                    lose_bank -= self.dynamic_bet
                    lose_bets += 1

                total_coef += accepted_coef
                self.detail_static_bank_values.append(self.current_static_bank)

            else:
                skipped_bets += 1

        try:
            average_win_coef = win_coef / win_bets
        except ZeroDivisionError:
            average_win_coef = 0

        self.batch_static_bank_values.append(self.current_static_bank)
        try:
            percent_profit = (self.current_static_bank - initial_bank) / initial_bank * 100
        except ZeroDivisionError:
            percent_profit = 0
        try:
            average_lose_coef = lose_coef / lose_bets
        except ZeroDivisionError:
            average_lose_coef = 0
        try:
            average_coef = total_coef / accepted_bets
        except ZeroDivisionError:
            average_coef = 0

        result = {'skipped_bets': skipped_bets,
                  'accepted_bets': accepted_bets,
                  'profit': total_profit,
                  'percent_profit': percent_profit,
                  'initial_bank': initial_bank,
                  'total_bank': self.current_static_bank,
                  'win_bank': win_bank,
                  'lose_bank': lose_bank,
                  'win_bets': win_bets,
                  'lose_bets': lose_bets,
                  'average_coef': average_coef,
                  'average_win_coef': average_win_coef,
                  'average_lose_coef': average_lose_coef,
                  'win_rate': win_bets / accepted_bets if accepted_bets > 0 else 0
                  }
        print(self.current_static_bank)
        print(result)
        self.dynamic_bet = self.current_static_bank * 0.01
        print('bet', self.dynamic_bet)
        self.full_static_info.append(result)

    def update_static_bank(self, bank: float):
        """ Count new bank with static bet"""

        self.current_static_bank = bank

    def update_dynamic_bank(self, bank: float):
        """ Count new bank using % of current bank as bet """

        self.current_dynamic_bank = bank
        self.update_dynamic_bet()

    def update_dynamic_bet(self):
        self.dynamic_bet = self.current_dynamic_bank * self.percent_of_bank
