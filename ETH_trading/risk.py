import pandas as pd
import numpy as np

from app import diary

max_trade_risk = 0.01
max_open_risk = 0.05
commission = 0.001

leverage_factor = 5  # this is either 5 or 3 on BINANCE


""" Functions to do calculations """


def position_size(cap, buy, stop, leverage=False):
    """ Calculate the allowed position size

    Note: Works for both SHORT and LONG trades. Be careful to use the right input order!

    If user chooses to trade with leverage, position size is limited to 5x capital
    """

    high = max(buy, stop)
    low = min(buy, stop)

    max_cap_risk = cap*max_trade_risk
    true_size = max_cap_risk/abs(high/(1-commission) - low*(1-commission))

    if buy > stop:
        # Max size for long trade:
        max_size = cap/high
    else:
        # Max size for short:
        max_size = cap/low

    if leverage:
        max_size = max_size*leverage_factor

    return min(true_size, max_size)


def trade_risk(cap, trades):
    """ Calculate the risk (%) of (an array of) open trades

    INPUT: cap is the current capital, trades is a df with !open! trades
    """

    buy = trades['buy'].to_numpy()
    size = trades['size'].to_numpy()
    stop = trades['stop'].to_numpy()

    high = np.maximum(buy, stop)
    low = np.minimum(buy, stop)

    loss = size*abs(high/(1-commission) - low*(1-commission))
    risk = loss/cap*100

    return risk


def open_profit(trades):
    # GET current pair prices from BINANCE
    return 1