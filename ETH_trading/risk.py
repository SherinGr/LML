import pandas as pd
import numpy as np
import shelve
import datetime

#import cryptowatch as cw

from app import client  #, user_data

import tabs.close as closetab

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


def profit_abs(entry, close, qty, direction):
    if direction == 'SHORT':
        return qty*(entry*(1-commission) - close*(1+commission))
    elif direction == 'LONG':
        return qty*(close*(1-commission)**2 - entry)
    else:
        raise NameError('I do not recognize the trade direction "' + direction + '"')


def profit_rel(entry, close, qty, direction):
    cap = user_data['capital'][-1]
    p_abs = profit_abs(entry, close, qty, direction)
    p_rel = p_abs/cap*100
    return p_rel


def risk_reward_ratio(entry, stop, close):
    if close == stop:
        return 0
    else:
        return abs((close-entry)/(entry-stop))


def update_shelf(varname, new_data):
    # TODO: May not be necessary to open and close the shelve in these functions.
    """" Update a variable in the shelf as a moving average

    INPUT: new_point is a pd.Series with datetimeIndex
    """
    with shelve.open('user_data') as d:
        # Retrieve the old win rate and amount of recorded points:
        data = d[varname]
        old_value = data[-1]
        n = len(data)
        # Calculate the new win rate:
        new_value = (n*old_value + new_data)/(n+1)
        # Add the result to the user_data:
        d[varname] = data.append(new_value)

    return new_value


def update_win_rate(trade):
    # Check if the trade is a winner:
    trade = trade.set_index('date')
    is_winner = trade['P/L (%)'] > 0

    new_win_rate = update_shelf('win_rate', is_winner)

    return new_win_rate


def update_avg_profit(trade):
    t = trade.iloc[0]
    p_rel = t['P/L (%)']

    df = pd.Series([p_rel], pd.DatetimeIndex([t['date']]))
    new_avg_profit = update_shelf('avg_profit', df)

    return new_avg_profit


def update_avg_rrr(trade):
    t = trade.iloc[0]
    entry = t['entry']
    stop = t['stop']
    close = t['exit']

    if not close == stop:
        rrr = risk_reward_ratio(entry, stop, close)
        new_avg_rrr = update_shelf('avg_rrr', rrr)
    else:
        new_avg_rrr = 0

    return new_avg_rrr


def update_avg_timespan(trade):
    # TODO: Not 100% sure if adding and averaging times works, check it!
    t = trade.iloc[0]
    delta = t['timespan']

    df = pd.Series([delta], pd.DatetimeIndex([t['date']]))
    new_avg_timespan = update_shelf('avg_timespan', df)

    return new_avg_timespan


def update_expectancy(trade):
    # TODO: Make this function.
    #   E = (1 + avg_win/avg_loss) * win_rate - 1
    pass


def close_trade(open_trade, close, note):
    """" Fill in the remaining values from an open trade into a closed trade dataFrame"""
    index = pd.DatetimeIndex([datetime.datetime.now()])
    cols = closetab.closed_trade_cols
    closed_trade = pd.DataFrame(index=index, columns=cols)

    # Copy values that already exist.
    for c in open_trade.columns:
        closed_trade[c] = open_trade[c]
    # TODO: Fill in the remaining values
    closed_trade['P/L (%)'] = profit_rel(entry, close, qty, direction)
    closed_trade['risk (%)'] = trade_risk(cap, open_trade)
    closed_trade['RRR'] = 0
    closed_trade['cap. share (%)'] = 0
    closed_trade['timespan'] = 0
    closed_trade['note'] = note

    return closed_trade


if __name__ == '__main__':
    trades = pd.read_excel('diary.xlsx', sheet_name='closed')

    user_data = shelve.open('user_data')
    cap = user_data['capital'][-1]
    x = trades.tail(1)

    risk = trade_risk(cap, x)

    wr = update_win_rate(x)
    avg_prof = update_avg_profit(x)
    avg_rrr = update_avg_rrr(x)

    user_data = shelve.open('user_data')
    p = user_data['avg_profit']
    wr = user_data['win_rate']
    r = user_data['avg_rrr']


