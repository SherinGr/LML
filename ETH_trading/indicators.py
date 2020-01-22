# This file defines the indicators that can be used in a trading strategy
#
# In case you are interested in a more elaborate library check out tulipy

import numpy as np
import pandas as pd
import warnings
from scipy.linalg import toeplitz
from collections import deque


class Indicator:
    def __init__(self, window_length, time_frame):
        self.window_length = window_length
        self.time_frame = time_frame
        # Initialize candles in memory and history of the MA:
        self.memory = deque([np.nan] * window_length, maxlen=self.window_length)  # candle aspects inside the window
        self.history = pd.DataFrame()                                           # history of all values calculated


class MovingAverage(Indicator):
    def __init__(self, window_length, time_frame):
        super().__init__(window_length, time_frame)

    def update(self, new_data):
        # TODO: For a later version, only update if the time_frame matches
        try:
            # Add the new candle close to the memory:
            self.memory.append(new_data['close'])
            # Calculate the MA from the new deque:
            new_value = np.nansum(list(self.memory)) / self.window_length

            df = pd.DataFrame(new_value)
            df.index = pd.DataFrame([new_data.name])
        except TypeError:
            # If input is just a single value, do standard MA
            self.memory.append(new_data)
            # Calculate the MA from the new deque:
            new_value = np.nansum(list(self.memory))/self.window_length

            df = pd.DataFrame([new_value])
            # Make sure the indexing of the df is correct:
            if self.history.empty:
                df.index = pd.Int64Index([0])
            else:
                df.index = self.history.tail(1).index + 1

        self.history = self.history.append(df)

        return new_value

    def batch_fit(self, data_batch):
        """ Fit the moving average of a batch of candles"""
        if not self.history.empty:
            self.history = []
            self.memory = deque([np.nan] * window_length, maxlen=self.window_length)
            warnings.warn('Old MA data has been removed! Make sure that this was your intention', UserWarning)

        try:
            # Assume candle input
            self.history = data_batch['close'].rolling(self.window_length).mean()
            self.memory.extend(data_batch['close'].tail(self.window_length))
        except TypeError:
            # Otherwise assume list or array of values
            df = pd.DataFrame(data_batch)
            self.history = df.rolling(self.window_length).mean()
            self.memory.extend(df.tail(self.window_length))


class ExponentialMovingAverage(Indicator):
    def __init__(self, window_length, time_frame):
        super().__init__(window_length, time_frame)

        self.coefficient = 2/(window_length+1)

    def new_ema(self, new_data):
        try:
            # If we get a candle, add the new candle close to the memory:
            self.memory.append(new_data['close'])
        except (TypeError, IndexError):
            # If we get a scalar value, add it to the memory as such:
            self.memory.append(new_data)

        # Get the previous EMA value:
        if self.history.empty:
            old_ema = self.memory[-1]
        else:
            old_ema = self.history.tail(1)

        # Calculate the new EMA value:
        new_ema = self.memory[-1] * self.coefficient + old_ema * (1 - self.coefficient)

        return new_ema

    def update(self, new_data):
        new_value = self.new_ema(new_data)
        df = pd.DataFrame([new_value])
        # Setting the index of the df:
        try:
            # Add new value to the history DataFrame with DateTimeIndex:
            df.index = pd.DatetimeIndex([new_data.name])
        except AttributeError:
            # If new_data was not candle:
            if self.history.empty:
                df.index = pd.Int64Index([0])
            else:
                df.index = self.history.tail(1).index + 1
        # Add new value to the history df:
        self.history = self.history.append(df)

        return new_value

    def batch_fit(self, data_batch):
        if not self.history.empty:
            self.history = []
            self.memory = deque([np.nan] * window_length, maxlen=self.window_length)
            warnings.warn('Old EMA data has been removed! Make sure that this was your intention.', UserWarning)

        try:
            self.history = data_batch['close'].ewm(span=self.window_length).mean()
            self.memory.extend(data_batch['close'].tail(self.window_length))
        except TypeError:
            df = pd.DataFrame(data_batch)
            self.history = df.ewm(span=self.window_length).mean()
            self.memory.extend(df.tail(self.window_length))


class ATR(Indicator):
    def __init__(self, window_length, time_frame):
        super().__init__(window_length, time_frame)
        self.ema = ExponentialMovingAverage(window_length, time_frame)

    def true_range(self, candle):
        prev_candle = self.memory[-1]

        try:
            tr = max(candle['high'] - candle['low'],
                     abs(candle['high'] - prev_candle['close']),
                     abs(candle['low'] - prev_candle['close']))
        except TypeError:
            # If there is no previous candle, use only current candle
            tr = candle['high'] - candle['low']

        return tr

    def update(self, candle):
        new_tr = self.true_range(candle)
        new_atr = self.ema.update(new_tr)

        self.memory.append(candle)  # don't put this in front of the true_range fn!

        df = pd.DataFrame([new_atr])
        df.index = pd.DatetimeIndex([candle.name])
        self.history = self.history.append(df)

        return new_atr

    def batch_fit(self, candle_batch):
        if not self.history.empty:
            self.history = pd.DataFrame()
            self.memory = deque([np.nan] * window_length, maxlen=self.window_length)

        batch_size = len(candle_batch)
        print('Sorry, encountered a slow for loop...')

        for i in range(batch_size):
            candle = candle_batch.iloc[i]
            self.update(candle)

        # Add last #window_length candles to the memory
        self.memory.extend(candle_batch['close'].tail(self.window_length))


class ATRChannels:
    def __init__(self, window_length, time_frame):
        self.atr = ATR(window_length, time_frame)
        self.ema = ExponentialMovingAverage(window_length, time_frame)

        self.atr_bands = pd.DataFrame()

    def update(self, candle):
        new_atr = self.atr.update(candle)
        new_ema = self.ema.update(candle)

        new_bands = self.atr_channels(new_ema, new_atr)

        df = pd.DataFrame([new_bands])
        df.index = pd.DatetimeIndex([candle.name])

        self.atr_bands.append(df)

        return new_bands

    @staticmethod
    def atr_channels(ema, atr):
        """ Get -3,-2,-1,+1,+2,+3 ATR array around EMA"""
        ema = np.array(ema)
        atr = np.array(atr)
        ranges = np.array([[3], [2], [1], [0], [-1], [-2], [-3]])
        atr_bands = ranges * atr.T + np.ones(ranges.shape) * ema
        # TODO: shapes go wrong with the batch fit.
        return atr_bands

    def batch_fit(self, candle_batch):
        # Calculate the ATR and the EMA over all candles
        self.atr.batch_fit(candle_batch)
        self.ema.batch_fit(candle_batch)
        # Construct the ATR bands around the EMA:
        atr_bands = self.atr_channels(self.ema.history, self.atr.history)
        # Save the data in a DataFrame
        df = pd.DataFrame(atr_bands.T)
        df.index = pd.DatetimeIndex(candle_batch.index)
        self.atr_bands = df


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
        self.memory = deque([np.nan]*(window_length+1), maxlen=window_length+1)
        self.history = []

        self.average_gain = 0
        self.average_loss = 0

    def update(self, new_candle):
        new_close = new_candle['close']
        print(new_candle)
        self.memory.append(new_close)
        # Calculate all changes inside the window:
        close_values = np.array(self.memory)
        change = close_values[-2]-close_values[-1]
        # Calculate gains and losses (0 if change is not the right sign)
        gain = change*(change > 0)
        loss = change*(change < 0)
        # Average gain and loss over window:
        self.average_gain = (self.average_gain*(self.window_length-1)+gain)/self.window_length
        self.average_loss = (self.average_loss*(self.window_length-1)+abs(loss))/self.window_length
        # Calculate RSI:
        try:
            relative_strength = self.average_gain/self.average_loss
            new_rsi = 100 - 100 / (1 + relative_strength)
        except ZeroDivisionError:
            new_rsi = 100

        # TODO: An EMA needs to be added over this

        self.history = self.history + [new_rsi]

    def batch_fit(self, candle_batch):
        if self.history:
            self.history = []
            self.memory = deque([np.nan] * (window_length + 1), maxlen=self.window_length + 1)
            warnings.warn('Old RSI data has been removed! Make sure that this was your intention', UserWarning)

        for candle in candle_batch.iterrows():
            print(candle['close'])
            self.update(candle)


class StochasticRSI(RSI):
    pass

