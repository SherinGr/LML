import pandas as pd
import numpy as np
import shelve

import cryptowatch as cw

from app import user_data, client

max_open_risk = 0.05
commission = 0.001

# TODO: RENAME THIS MODULE

""" Functions to do calculations """
# This module has functions to calculate features of a trade, such as the maximum size, risk, profit, etc.


def max_qty(entry, stop, max_risk, leverage=1):
    """ Calculate the allowed position size

    Note: Works for both SHORT and LONG trades. Be careful to use the right input order!

    If user chooses to trade with leverage, position size is limited to 5x capital
    """

    cap = user_data['capital'][-1]
    max_cap_risk = -cap*max_risk
    qty_cap = cap/entry*leverage                 # max quantity limited by capital

    # max quantity by risk:
    if entry > stop:    # LONG
        qty_risk = max_cap_risk/(stop*(1-commission)**2-entry)
    else:               # SHORT
        qty_risk = max_cap_risk/(entry*(1-commission) - stop*(1+commission))

    return min(qty_risk, qty_cap)


def trade_risk(cap, trades):
    """ Calculate the risk (%) of (an array of) open trades

    INPUT: cap is the current capital, trades is a df with !open! trades
    """

    entry = trades['entry'].to_numpy()
    size = trades['size'].to_numpy()
    stop = trades['stop'].to_numpy()

    high = np.maximum(entry, stop)
    low = np.minimum(entry, stop)

    # NOTE: This formula is an approximation! Now we can use one equation for short and long trades.
    loss = size*abs(high/(1-commission) - low*(1-commission))
    risk = loss/cap*100

    return risk


def open_profit(trades):

    # TODO: This for loop makes calls to binance API, so execution speed depends on how many calls per second binance
    #  allows us to make. Better to save a current price somewhere that is updated every x seconds.

    current_price = pd.Series([])
    for s in trades['pair']:
        ticker = client.get_symbol_ticker(symbol=s)
        cp = float(ticker['price'])
        current_price = current_price.append(pd.Series(cp), ignore_index=True)

    size = trades['size']
    entry = trades['entry']

    open_value = size.multiply(current_price*(1-commission) - entry)

    return sum(open_value)


def profit_abs(entry, exit, qty, direction):
    if direction == 'SHORT':
        return qty*(entry*(1-commission) - exit*(1+commission))
    elif direction == 'LONG':
        return qty*(exit*(1-commission)**2 - entry)
    else:
        raise NameError('I do not recognize the trade direction "' + direction + '"')


def profit_rel(entry, exit, qty, direction):
    cap = user_data['capital'][-1]
    p_abs = profit_abs(entry, exit, qty, direction)
    p_rel = p_abs/cap*100
    return p_rel


def risk_reward_ratio(entry, stop, exit):
    return abs((exit-entry)/(entry-stop))


def update_expectancy():
    pass


def update_shelf(entry, new_point):
    # TODO: May not be necessary to open and close the shelve in these functions.
    """" Update a variable in the shelf as a moving average

    INPUT: new_point is a pd.Series with datetimeIndex
    """
    with shelve.open('user_data') as user_data:
        # Retrieve the old win rate and amount of recorded points:
        data = user_data[entry]
        old_value = data[-1]
        n = len(data)
        # Calculate the new win rate:
        new_value = (n*old_value + new_point)/(n+1)
        # Add the result to the user_data:
        user_data[entry] = data.append(new_value)

    return new_value


def update_win_rate(trade):
    # Check if the trade is a winner:
    trade = trade.set_index('date')
    is_winner = trade['P/L (%)'] > 0

    new_win_rate = update_shelf('win_rate', is_winner)

    return new_win_rate


def update_avg_profit(trade):
    trade = trade.iloc[0]
    p_rel = trade['P/L (%)']

    df = pd.Series([p_rel], pd.DatetimeIndex([trade['date']]))
    new_avg_profit = update_shelf('avg_profit', df)

    return new_avg_profit


def update_avg_rrr(trade):
    trade = trade.iloc[0]
    entry = trade['entry']
    stop = trade['stop']
    exit = trade['exit']

    rrr = risk_reward_ratio(entry, stop, exit)
    # TODO: Do not update if it is a loss!
    new_avg_rrr = update_shelf('avg_rrr', rrr)

    return new_avg_rrr


def update_avg_timespan(trade):
    trade = trade.iloc[0]

    pass


def close_trade(open_trade):
    # fill in the remaining variables, return complete dataFrame.
    pass


if __name__ == '__main__':
    trades = pd.read_excel('diary.xlsx', sheet_name='closed')

    x = trades.tail(1)
    wr = update_win_rate(x)
    avg_prof = update_avg_profit(x)
    avg_rrr = update_avg_rrr(x)

    user_data = shelve.open('user_data')
    p = user_data['avg_profit']
    wr = user_data['win_rate']
    r = user_data['avg_rrr']


