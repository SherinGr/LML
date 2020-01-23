# This file defines the indicators that can be used in a trading strategy
#
# In case you are interested in a more elaborate library check out tulipy

import numpy as np
import pandas as pd
import warnings
import plotly.graph_objects as go
from scipy.linalg import toeplitz
from scipy.linalg import hankel
from collections import deque


class Indicator:
    def __init__(self, window_length, time_frame, tp_style):
        self.window_length = window_length
        self.time_frame = time_frame
        # Initialize candles in memory and history of the MA:
        self.memory = deque([np.nan] * window_length, maxlen=self.window_length)  # candle aspects inside the window
        self.history = pd.DataFrame()     # history of all values calculated
        # Set the typical price used to base the indicator on: (see fn below for options)
        self.tp_style = tp_style

    @staticmethod
    def get_tp(candles, tp_style):
        """ Get the typical price of a candle batch for given tp_style

            :returns int if one candle (pd.Series) as input, otherwise returns pd.DataFrame
        """
        if tp_style in ['open', 'high', 'low', 'close']:
            return candles[tp_style]
        elif tp_style == 'hl2':
            return (candles['high'] + candles['low'])/2
        elif tp_style == 'hlc3':
            return (candles['high'] + candles['low'] + candles['close'])/3
        elif tp_style == 'ohlc4':
            if len(candles.shape) == 1:
                # If only one candle it is a pd.Series so we have to do this:
                return candles.drop('volume base').mean()
            else:
                return candles.drop(columns=['volume base']).mean(axis=1)
        else:
            raise NameError('"' + tp_style + '" is not recognized as a typical price format.')


class MovingAverage(Indicator):
    def __init__(self, window_length, time_frame, tp_style='close'):
        super().__init__(window_length, time_frame, tp_style)
        self.name = 'MA' + str(window_length) + '_' + time_frame

    def update(self, new_data):
        # TODO: For a later version, only update if the time_frame matches
        try:
            # Calculate new MA for a candle input:
            new_tp = super().get_tp(new_data, self.tp_style)
            new_ma = np.nansum(list(self.memory)) / self.window_length

            self.memory.append(new_tp)
            df = pd.DataFrame(new_ma)
            df.index = pd.DatetimeIndex([new_data.name])
        except TypeError:
            # If input is just a single value, do standard MA
            self.memory.append(new_data)
            # Calculate the MA from the new deque:
            new_ma = np.nansum(list(self.memory))/self.window_length

            df = pd.DataFrame([new_ma])
            # Make sure the indexing of the df is correct:
            if self.history.empty:
                df.index = pd.Int64Index([0])
            else:
                df.index = self.history.tail(1).index + 1

        self.history = self.history.append(df)

        return new_ma

    def batch_fit(self, data_batch):
        """ Fit the moving average of a batch of candles"""
        if not self.history.empty:
            self.history = []
            self.memory = deque([np.nan] * window_length, maxlen=self.window_length)
            warnings.warn('Old MA data has been removed! Make sure that this was your intention', UserWarning)

        try:
            # Assume candle input
            tp_batch = super().get_tp(data_batch, self.tp_style)
            self.history = tp_batch.rolling(self.window_length).mean()
            self.memory.extend(tp_batch.tail(self.window_length))
        except IndexError:
            # Otherwise assume list or array of values
            df = pd.DataFrame(data_batch)
            self.history = df.rolling(self.window_length).mean()
            self.memory.extend(df.tail(self.window_length))

    def plot(self, figure, color='purple'):
        ma = self.history
        t = self.history.index
        figure.add_trace(go.Scatter(x=t, y=ma, line=dict(color=color), name=self.name),
                         row=1, col=1)


class ExponentialMovingAverage(Indicator):
    def __init__(self, window_length, time_frame, tp_style='close'):
        super().__init__(window_length, time_frame, tp_style)
        self.name = 'EMA' + str(window_length) + '_' + time_frame

        self.coefficient = 2/(window_length+1)

    def new_ema(self, new_data):
        """ This function also makes the EMA class usable on scalar series"""
        if type(new_data) == np.float64 or type(new_data) == float:
            # If we get a scalar value, add it to the memory as such:
            self.memory.append(new_data)
        elif type(new_data) == pd.core.series.Series:
            # If we get a candle, add the new candle typical price to the memory:
            new_tp = super().get_tp(new_data, self.tp_style)
            self.memory.append(new_tp)
        else:
            raise TypeError('Wrong data type {} supplied to ema update'.format(type(new_data)))

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
            tp_batch = super().get_tp(data_batch, self.tp_style)
            self.history = tp_batch.ewm(span=self.window_length).mean()
            self.memory.extend(tp_batch.tail(self.window_length))
        except NameError:
            df = pd.DataFrame(data_batch)
            self.history = df.ewm(span=self.window_length).mean()
            self.memory.extend(df.tail(self.window_length))

    def plot(self, figure, color='purple'):
        ema = self.history
        t = ema.index
        figure.add_trace(go.Scatter(x=t, y=ema, line=dict(color=color), name=self.name),
                         row=1, col=1)


class ATR(Indicator):
    def __init__(self, window_length, time_frame):
        super().__init__(window_length, time_frame, tp_style=None)
        self.name = 'ATR' + str(window_length) + '_' + time_frame

        self.ema = ExponentialMovingAverage(window_length, time_frame, tp_style='other')

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
            self.ema.history = pd.DataFrame()
            self.memory = deque([np.nan] * window_length, maxlen=self.window_length)
            warnings.warn('Old ART data has been removed. Make sure this was your intention', UserWarning)

        # First case of TR: Ht - Lt
        v1 = np.array(candle_batch['high']) - np.array(candle_batch['low'])
        # Second case of TR: Ct - Ct-1
        temp = np.append(np.nan, np.array(candle_batch.iloc[:-1]['close']))
        v2 = abs(np.array(candle_batch['close']) - temp)
        # Third case of TR: Lt - Ct-1
        v3 = abs(np.array(candle_batch['low']) - temp)

        v = np.array([v1, v2, v3])
        tr_array = np.max(v, axis=0)

        self.ema.batch_fit(tr_array)
        self.history = self.ema.history

        # Add last #window_length candles to the memory
        self.memory.extend(candle_batch['close'].tail(self.window_length))


class ATRChannels:
    def __init__(self, window_length, time_frame):
        self.atr = ATR(window_length, time_frame)
        self.ema = ExponentialMovingAverage(window_length, time_frame)

        self.channels = pd.DataFrame()

    def update(self, candle):
        # Update the data:
        new_atr = self.atr.update(candle)
        new_ema = self.ema.update(candle)
        new_bands = self.atr_channels(new_ema, new_atr)
        # Store the new data:
        cols = ['+3', '+2', '+1', 'EMA', '-1', '-2', '-3']
        temp = dict(zip(cols, new_bands))
        index = pd.DatetimeIndex([candle.name])
        df = pd.DataFrame.from_records(temp, index=index)
        self.channels.append(df)

        return new_bands

    @staticmethod
    def atr_channels(ema, atr):
        """ Get -3,-2,-1,+1,+2,+3 ATR array around EMA"""
        ema = np.array(ema)
        atr = np.array(atr)
        ranges = np.array([[3], [2], [1], [0], [-1], [-2], [-3]])
        channels = ranges * atr.T + np.ones(ranges.shape) * ema
        return channels

    def batch_fit(self, candle_batch):
        # Calculate the ATR and the EMA over all candles
        self.atr.batch_fit(candle_batch)
        self.ema.batch_fit(candle_batch)
        # Construct the ATR bands around the EMA:
        channels = self.atr_channels(self.ema.history, self.atr.history)
        # Save the data in a DataFrame
        df = pd.DataFrame(channels.T)
        df.index = pd.DatetimeIndex(candle_batch.index)
        df.columns = ['+3', '+2', '+1', 'EMA', '-1', '-2', '-3']
        self.channels = df

    def plot(self, figure):
        """ Plot +/-2 and +/-3 ATR around corresponding EMA"""
        channels = self.channels
        ema = self.ema
        t = channels.index
        # Plot the EMA:
        figure.add_trace(go.Scatter(x=t, y=ema.history, line=dict(color='orange'), name=ema.name),
                         row=1, col=1)

        # Drop columns that we do not want to plot up next:
        channels = channels.drop(columns=['+1', 'EMA', '-1'])
        # Plot +2,-2 thin, and +3,-3 thicker
        cols = channels.columns
        for c in cols:
            figure.add_trace(go.Scatter(x=t, y=channels[c], line=dict(color='black', width=(abs(int(c)) - 1) / 2),
                                        showlegend=False),
                             row=1, col=1)


class BollingerBand(Indicator):
    def __init__(self, window_length, time_frame, num_std=2, tp_style='close'):
        super().__init__(window_length, time_frame, tp_style)
        self.num_std = num_std
        self.tp_style = tp_style

        self.ma = MovingAverage(window_length, time_frame, self.tp_style)

    def get_std(self):
        # No, don't get an STD, just find the standard deviation of the last tp's!
        current_ma = self.ma.history.iloc[-1]
        tp_array = np.array(self.ma.memory)
        std = np.nanstd(tp_array - current_ma)

        return std

    def update(self, candle):
        # Update the MA (and its history as such):
        ma = self.ma.update(candle)
        std = self.get_std()  # this should not be called before updating the MA!
        upper_bb = ma + 2*std
        lower_bb = ma - 2*std
        # Store the data:
        bands = [upper_bb, ma, lower_bb]
        cols = ['UB', 'MA', 'LB']
        index = pd.DatetimeIndex([candle.name])
        df = pd.DataFrame.from_records(dict(zip(cols, bands)), index=index)
        self.history = self.history.append(df)
        self.memory = self.ma.memory

    def batch_fit(self, candle_batch):
        if not self.history.empty:
            self.history = []
            self.memory = deque([np.nan] * window_length, maxlen=self.window_length)
            warnings.warn('Old BB data has been removed! Make sure that this was your intention.', UserWarning)

        tp_batch = np.array(super().get_tp(candle_batch, self.tp_style))
        tp_matrix = hankel(tp_batch, tp_batch[-self.window_length:])

        self.ma.batch_fit(candle_batch)
        ma = np.array(self.ma.history)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # numpy complains that std has NaN's but this is OK for us.
            std = np.nanstd(tp_matrix.T - ma*np.ones((self.window_length, 1)), axis=0)

        upper_bb = ma + 2*std
        lower_bb = ma - 2*std

        bands = [upper_bb, ma, lower_bb]
        cols = ['UB', 'MA', 'LB']

        temp = dict(zip(cols, bands))
        df = pd.DataFrame.from_dict(temp)
        df.index = pd.DatetimeIndex(candle_batch.index)
        self.history = df
        self.memory.extend(tp_batch[-self.window_length:])

    def plot(self, figure):
        """ Plot Bollinger Bands around corresponding MA"""
        bands = self.history
        bands = bands.drop(columns=['MA'])
        t = bands.index

        # Plot the upper band:
        figure.add_trace(go.Scatter(x=t, y=bands['UB'], line=dict(color='black', width=0.5), showlegend=False,
                                    fill=None),
                         row=1, col=1)

        # Plot lower band and fill area:
        figure.add_trace(go.Scatter(x=t, y=bands['LB'], line=dict(color='black', width=0.5), showlegend=False,
                                    fill='tonexty', fillcolor='rgba(255,255,255,0.5)'),
                         row=1, col=1)


class MACD:
    pass


class RSI(Indicator):
    def __init__(self, time_frame, window_length=14, tp_style='close'):
        super().__init__(window_length, time_frame, tp_style)

        self.name = 'RSI' + str(window_length) + '_' + time_frame

        self.avg_gain = ExponentialMovingAverage(window_length, time_frame, tp_style)
        self.avg_loss = ExponentialMovingAverage(window_length, time_frame, tp_style)

    def update(self, candle):
        new_tp = super().get_tp(candle, self.tp_style)
        # Calculate all changes inside the window:
        change = new_tp - self.memory[-1]
        # Calculate gains and losses (0 if change is not the right sign)
        gain = change*(change > 0)
        loss = change*(change < 0)
        # Update average gain and loss EMA's:
        new_avg_gain = self.avg_gain.update(gain)
        new_avg_loss = self.avg_loss.update(loss)
        # TODO: check if this works?
        # self.average_gain = (self.average_gain*(self.window_length-1)+gain)/self.window_length
        # self.average_loss = (self.average_loss*(self.window_length-1)+abs(loss))/self.window_length
        # Calculate RSI:
        try:
            relative_strength = new_avg_gain/new_avg_loss
            new_rsi = 100 - 100 / (1 + relative_strength)
        except ZeroDivisionError:
            new_rsi = 100
        # TODO: start debugging here!
        df = pd.DataFrame([new_rsi])
        df.index = pd.DatetimeIndex([candle.name])
        self.history = self.history.append(df)
        self.memory.append(new_tp)

    def batch_fit(self, candle_batch):
        if self.history.empty:
            self.history = []
            self.memory = deque([np.nan] * (self.window_length + 1), maxlen=self.window_length)
            warnings.warn('Old RSI data has been removed! Make sure that this was your intention', UserWarning)

        batch_size = len(candle_batch)
        # TODO: Remove slow for loop!
        print('Sorry, encountered slow for loop...')
        for i in range(batch_size):
            candle = candle_batch.iloc[i]
            self.update(candle)

    def plot(self, figure, color='royalblue'):
        rsi = self.history
        t = rsi.index
        figure.add_trace(go.Scatter(x=t, y=rsi, line=dict(color=color, width=3), name=self.name),
                         row=2, col=1)

class StochasticRSI(RSI):
    pass

