# This file defines the indicators that can be used in a trading strategy
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
        self.history = [1,2]                                   # history of all MA values calculated

    def update(self, new_candle):
        """ Adds a new value to the moving average line

        INPUTS: new_candle must be a list or np.array with OHLC values

        TODO: For a later version, only update if the timeframe matches
        """
        # Add the new candle aspect to the memory:
        self.memory.appendleft(new_candle(self.aspect_dict[self.aspect]))
        # Calculate the MA from the new deque:
        new_value = sum(list(self.memory))/self.window_length
        # Add the new value to the MA history:
        self.history = [new_value] + self.history

        return new_value

    def batch_fit(self, candle_batch):
        """ Fit the moving average of a batch of candles"""
        # TODO: Write this function (use reshape and matrix to do it in one go!
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
        self.ema_history = [new_ema] + self.ema_history

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
    def __init__(self, window_length=14):
        self.window_length = window_length
        self.memory = deque([],maxlen=window_length+1)
        self.history = []

    def update(self, new_candle):
        new_close = new_candle[3]  # hard coded, close should be the third value following OHLC convention

        self.memory.appendleft(new_close)
        # Find the differences between each candle and the preceding one:
        close_values = np.array(self.memory)
        changes = close_values[:-1] - close_values[1:]
        # Calculate the total amount of up and down differences:
        up_sum = sum(changes*(changes>0))
        down_sum= abs(sum(changes*(changes<=0)))

        new_value = 100 - 100/(1 + up_sum/down_sum)
        self.history = [new_value] + self.history
        # TODO: Not 100% if not a second smoothing is needed


class StochasticRSI(RSI):
    pass
