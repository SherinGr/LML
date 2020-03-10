import pandas as pd
import numpy as np

import cryptowatch as cw

from app import user_data, client

max_open_risk = 0.05
commission = 0.001


""" Functions to do calculations """


def position_size(cap, entry, stop, max_risk, leverage=1):
    """ Calculate the allowed position size

    Note: Works for both SHORT and LONG trades. Be careful to use the right input order!

    If user chooses to trade with leverage, position size is limited to 5x capital
    """

    high = max(entry, stop)
    low = min(entry, stop)

    max_cap_risk = cap*max_risk
    true_size = max_cap_risk/abs(high/(1-commission) - low*(1-commission))

    if entry > stop:
        # Max size for long trade:
        max_size = cap/high
    else:
        # Max size for short:
        max_size = cap/low

    max_size = max_size*leverage

    return min(true_size, max_size)


def trade_risk(cap, trades):
    """ Calculate the risk (%) of (an array of) open trades

    INPUT: cap is the current capital, trades is a df with !open! trades
    """

    entry = trades['entry'].to_numpy()
    size = trades['size'].to_numpy()
    stop = trades['stop'].to_numpy()

    high = np.maximum(entry, stop)
    low = np.minimum(entry, stop)

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


if __name__ == '__main__':
    trades = pd.read_excel('diary.xlsx', sheet_name='open')
    trades = trades.drop(columns=['date'])

    value = open_profit(trades)
