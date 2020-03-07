# This file contains some functions to perform risk management:

max_trade_risk = 0.01
max_open_risk = 0.05
commission = 0.001


def trade_risk(capital, size, high, low):
    loss = size*abs(high/(1-commission) - low*(1-commission))
    risk = loss/capital
    return risk


def find_position_size(cap, high, low, direction):
    # VALID FOR BOTH LONG AND SHORT TRADES:
    # high is the high price of the trade (entry for LONG, stoploss for SHORT)
    # low is the low price of the trade (entry for SHORT, stoploss for LONG)
    max_cap_risk = cap*max_trade_risk
    true_size = max_cap_risk/abs(high/(1-commission) - low*(1-commission))

    if direction == 'long':
        max_size = cap/high
    elif direction == 'short':
        max_size = cap/low

    return min(true_size, max_size)


def get_open_risk():
    pass
    # look at all open trades and calculate open risk


if __name__ == '__main__':
    # LONG:
    entry = 239.18
    stoploss = 231.10
    capital = 100

    size = find_position_size(capital, entry, stoploss, direction='long')
    loss = entry*size - stoploss*size*(1-commission)**2
    risk = trade_risk(capital, size, entry, stoploss)

    # SHORT:
    entry = 239.18
    stoploss = 248.25
    capital = 100

    size2 = find_position_size(capital, stoploss, entry, direction='short')
    loss2 = size2*stoploss/(1-commission) - entry*size2*(1 - commission)
    risk2 = trade_risk(capital, size2, stoploss, entry)

