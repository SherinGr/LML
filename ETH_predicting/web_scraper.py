# We will webscrape the data of ETH prices from cryptowat.ch
import cryptowatch as cw
import pandas as pd

from datetime import datetime

# For plotting candlesticks:
import plotly.graph_objects as go
# And to render interactive plots from plotly in your browser:
import plotly.io as pio
pio.renderers.default = "browser"


def load_data():
    """ This function webscrapes data of ETH prices on BINANCE. As it is now it returns 4h OHCL and volume data. This
    can be changed according to your needs.

    :return: candles_4h
    """

    print('Fetching data\n')
    data = cw.markets.get("BINANCE:ETHUSDT", ohlc=True, periods=['5m', '15m', '1h', '4h', '1d'])
    # This is where the web scraping has to be done with a new (larger) dataset.

    print("Number of 5min candles:", len(data.of_5m))
    print("Number of 15min candles:", len(data.of_15m))
    print("Number of 1h candles:", len(data.of_1h))
    print("Number of 4h candles:", len(data.of_4h))
    print("Number of 1d candles:", len(data.of_1d))

    # The data is a list of lists, each element of the list is a list containing the following values:
    cols = ['close_timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_quote']

    # We will put the 4h data in a pandas dataframe:
    candles_4h = pd.DataFrame(data.of_4h, columns=cols)

    # We remove the volume_quote column, since it provides no additional information. We also remove the timestamp since
    # absolute time does not matter, only relative time, we store the times however, they come in handy later.
    close_stamps = candles_4h.close_timestamp
    candles_4h.drop(columns=['close_timestamp', 'volume_quote'])

    return candles_4h

# Let us plot some candles to see what we are dealing with:
#fig = go.Figure(data=[go.Candlestick(x=datetime.fromtimestamp(close_stamps),
#                                     open=candles_4h.open,
#                                     high=candles_4h.high,
#                                     low=candles_4h.low,
#                                     close=candles_4h.close)])
#fig.show()


class BatchGenerator:
    """This class generates a batch of data to be used as input for the LSTM network"""
    def __init__(self, data, num_candles, batch_size):
        self.data = data
        self.num_candles = num_candles
        self.batch_size = batch_size

    def generate_batch(self):
        pass





