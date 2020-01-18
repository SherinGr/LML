# We will webscrape the data of ETH prices from cryptowat.ch
import cryptowatch as cw
import pandas as pd


def load_data(time_frame='4h'):
    """ This function webscrapes data of ETH prices on BINANCE. As it is now it returns 4h OHCL and volume data. This
    can be changed according to your needs.

    :return: candles_4h
    """

    print('Fetching data\n')
    data = cw.markets.get("BINANCE:ETHUSDT", ohlc=True, periods=[time_frame])
    # This is where the web scraping has to be done with a new (larger) dataset.

    # Reference to the candle data in string format:
    ref = 'data.of_' + time_frame

    print("Number of {} candles:".format(time_frame), len(eval(ref)))

    # The data is a list of lists, each element of the list is a list containing the following values:
    cols = data._legend

    # We will put the 4h data in a pandas dataframe:
    candles = pd.DataFrame(eval(ref), columns=cols)

    # We remove the volume_quote column, since it provides no additional information. We also remove the timestamp since
    # absolute time does not matter, only relative time:
    candles = candles.drop(columns=['volume quote'])

    return candles



