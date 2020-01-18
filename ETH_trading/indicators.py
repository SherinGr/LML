# This file defines the indicators that can be used in a trading strategy
#
# In case you are interested in a more elaborate library check out tulipy

import numpy as np
import warnings
from scipy.linalg import toeplitz
from collections import deque


class MovingAverage:
    """ Constructs a moving average with a certain window length and lag"""
    def __init__(self, window_length, time_frame, aspect='C'):
        """
        Arguments:
        window_length: amount of candles to use in calculation
        time_frame: string denoting the timeframe (e.g. "4h")
        aspect: string which can be either of these four: OHLC (default C)
        """
        self.window_length = window_length
        self.time_frame = time_frame
        self.aspect = aspect.upper()

        # To know which element to get from the candle based on the aspect input:
        self.aspect_dict = {'O': 0, 'H': 1, 'L': 2, 'C': 3}

        self.memory = deque([], maxlen=self.window_length)  # candle aspects inside the window
        self.history = []                                   # history of all MA values calculated

    def update(self, new_candle):
        """ Adds a new value to the moving average line

        INPUTS: new_candle must be a list or np.array with OHLC values

        TODO: For a later version, only update if the time_frame matches
        TODO: For a later version, change input to pandas.df to robustly verify which aspect you extract
        """
        new_candle = np.array(new_candle)

        # Add the new candle aspect to the memory:
        self.memory.append(new_candle[self.aspect_dict[self.aspect]])
        # Calculate the MA from the new deque:
        new_value = sum(list(self.memory))/self.window_length
        # Add the new value to the MA history:
        self.history = self.history + [new_value]

        return new_value

    def batch_fit(self, candle_batch):
        """ Fit the moving average of a batch of candles"""
        candle_batch = np.array(candle_batch)  # in case we get a dataFrame input

        if self.history:
            self.history = []
            warnings.warn('Old MA data has been removed! Make sure that this was your intention', UserWarning)

        # Extract the price aspect of all candles (Close by default):
        prices = candle_batch[:, self.aspect_dict[self.aspect]]
        # Make toeplitz matrix for fast computation:
        r = prices[:self.window_length]
        c = prices[self.window_length:]
        price_matrix = toeplitz(c, r)
        self.history = np.ndarray.sum(price_matrix, axis=1)/self.window_length


class ExponentialMovingAverage(MovingAverage):
    def __init__(self, window_length, time_frame, aspect='C'):
        super().__init__(window_length, time_frame, aspect)

        self.coefficient = 2/(window_length+1)
        self.ema_history = []

    def update(self, new_candle):
        # First calculate the standard MA:
        new_ma = MovingAverage.update(self, new_candle)
        # New EMA value:
        new_ema = new_ma*self.coefficient + self.ema_history[0]*(1-self.coefficient)
        self.ema_history = self.ema_history + [new_ema]

    def batch_fit(self, candle_batch):
        # TODO: write this
        pass


class BollingerBands:
    def __init__(self):
        pass

    def update_bands(self,):
        pass


class MACD:
    pass


class RSI:
    # TODO: Values do not match tradingview yet.
    def __init__(self, window_length=14):
        self.window_length = window_length
        self.memory = deque([0]*(window_length+1), maxlen=window_length+1)
        self.history = []

        self.average_gain = 0
        self.average_loss = 0

    def update(self, new_candle):
        # TODO: if you only accept pandas df input you can verify what is the close value
        new_close = new_candle[3]  # hard coded, close should be the third value following OHLC convention
        self.memory.append(new_close)

        close_values = np.array(self.memory)
        change = close_values[-2]-close_values[-1]

        # Calculate gain and loss (0 if change is not the right sign)
        gain = change*(change > 0)
        loss = change*(change < 0)

        self.average_gain = (self.average_gain*(self.window_length-1)+gain)/self.window_length
        self.average_loss = (self.average_loss*(self.window_length-1)+abs(loss))/self.window_length

        try:
            relative_strength = self.average_gain/self.average_loss
            new_rsi = 100 - 100 / (1 + relative_strength)
        except ZeroDivisionError:
            new_rsi = 100

        self.history = self.history + [new_rsi]

    def batch_fit(self, candle_batch):
        candle_batch = np.array(candle_batch)
        batch_size = candle_batch.shape[0]

        if self.history:
            self.history = []
            warnings.warn('Old RSI data has been removed! Make sure that this was your intention', UserWarning)

        # TODO: you can calculate it all go as well using toeplitzes etc.
        for i in range(batch_size):
            self.update(candle_batch[i, :])

class StochasticRSI(RSI):
    pass

