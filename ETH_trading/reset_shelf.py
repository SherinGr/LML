from binance.client import Client

import datetime
import pandas as pd
import shelve

user_data = shelve.open('user_data')

keys = list(user_data.keys())

# Making a fake trade diary and capital evolution for performance page:
capital_array = [200]
date_array = [datetime.datetime.now()]

capital_df = pd.Series(capital_array, date_array)

user_data['capital'] = capital_df
user_data['n_trades'] = 0
user_data['n_wins'] = 0
user_data['n_losses'] = 0
user_data['total_loss'] = 0
user_data['total_gain'] = 0
user_data['expectancy'] = pd.Series(0, pd.DatetimeIndex([datetime.datetime.now()]))
user_data['win_rate'] = pd.Series(0.5, pd.DatetimeIndex([datetime.datetime.now()]))
user_data['avg_profit'] = pd.Series(0, pd.DatetimeIndex([datetime.datetime.now()]))
user_data['avg_rrr'] = pd.Series(1, pd.DatetimeIndex([datetime.datetime.now()]))
user_data['avg_timespan'] = pd.Series(60, pd.DatetimeIndex([datetime.datetime.now()]))


user_data.close()

