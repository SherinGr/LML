from binance.client import Client

import datetime
import pandas as pd
import shelve

user_data = shelve.open('user_data')

keys = list(user_data.keys())

# Making a fake trade diary and capital evolution for performance page:
capital_array = [221, 223.2, 222.51, 223.82, 230.53, 229.67, 228.52, 228.93, 221]
date_array = [datetime.datetime(2019, 12, 23, 13, 2),
              datetime.datetime(2019, 12, 24, 12, 48),
              datetime.datetime(2019, 12, 25, 9, 48),
              datetime.datetime(2019, 12, 25, 16, 7),
              datetime.datetime(2019, 12, 26, 17, 39),
              datetime.datetime(2019, 12, 27, 11, 25),
              datetime.datetime(2019, 12, 27, 23, 13),
              datetime.datetime(2019, 12, 28, 8, 29),
              datetime.datetime(2019, 12, 29, 13, 20)]

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
user_data['avg_timespan'] = pd.Series(datetime.timedelta(0), pd.DatetimeIndex([datetime.datetime.now()]))


user_data.close()

