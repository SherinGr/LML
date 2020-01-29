# This file defines the indicators that can be used in a trading strategy
#
# In case you are interested in a more elaborate library check out tulipy

import numpy as np
import pandas as pd
import warnings
import plotly.graph_objects as go
from scipy.linalg import hankel
from collections import deque

# TODO: time_frame attribute of classes is at this point obsolete. Remove or keep for later?


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

        self.name = 'MA_{}_{}'.format(self.window_length, self.tp_style)

    def update(self, new_data):
        if type(new_data) == pd.core.series.Series:
            # Calculate new MA for a candle input:
            new_tp = super().get_tp(new_data, self.tp_style)
            new_ma = np.nansum(list(self.memory)) / self.window_length

            self.memory.append(new_tp)
            df = pd.DataFrame(new_ma)
            df.index = pd.DatetimeIndex([new_data.name])
        else:
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

        if self.tp_style == 'other':
            # MA of list of values:
            df = pd.DataFrame(data_batch)
            self.history = df.rolling(self.window_length).mean()
            self.memory.extend(df.tail(self.window_length).values)
        else:
            # We get candles as input
            tp_batch = super().get_tp(data_batch, self.tp_style)
            self.history = tp_batch.rolling(self.window_length).mean()
            self.memory.extend(tp_batch.tail(self.window_length))

    def plot(self, figure, color='purple'):
        ma = self.history.values
        t = self.history.index
        figure.append_trace(go.Scatter(x=t, y=ma, line=dict(color=color), name=self.name),
                         row=1, col=1)


class ExponentialMovingAverage(Indicator):
    def __init__(self, window_length, time_frame, tp_style='close'):
        super().__init__(window_length, time_frame, tp_style)

        self.name = 'EMA_{}_{}'.format(self.window_length, self.tp_style)

        self.coefficient = 2/(window_length+1)

    def new_ema(self, new_data):
        """ This function also makes the EMA class usable on scalar series"""
        if self.tp_style == 'other':
            self.memory.append(new_data)
        else:
            new_tp = super().get_tp(new_data, self.tp_style)
            self.memory.append(new_tp)

        # Get the previous EMA value:
        if self.history.empty:
            old_ema = self.memory[-1]
        else:
            old_ema = self.history.tail(1).squeeze()

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

        if self.tp_style == 'other':
            df = pd.DataFrame(data_batch)
            self.history = df.ewm(span=self.window_length).mean()
            self.memory.extend(df.tail(self.window_length).values)
        else:
            tp_batch = super().get_tp(data_batch, self.tp_style)
            self.history = tp_batch.ewm(span=self.window_length).mean()
            self.memory.extend(tp_batch.tail(self.window_length))

    def plot(self, figure, color='purple'):
        ema = self.history.values
        t = self.history.index
        figure.append_trace(go.Scatter(x=t, y=ema, line=dict(color=color), name=self.name),
                         row=1, col=1)


class ATR(Indicator):
    def __init__(self, window_length, time_frame):
        super().__init__(window_length, time_frame, tp_style=None)
        self.ema = ExponentialMovingAverage(window_length, time_frame, tp_style='other')

        self.name = 'ATR_{}_{}'.format(self.window_length, self.ema.tp_style)

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

        self.name = 'ATR Channels_{}_{}'.format(self.ema.window_length, self.ema.tp_style)

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
        figure.append_trace(go.Scatter(x=t, y=ema.history, line=dict(color='orange'), name=ema.name),
                         row=1, col=1)

        # Drop columns that we do not want to plot up next:
        channels = channels.drop(columns=['+1', 'EMA', '-1'])
        # Plot +2,-2 thin, and +3,-3 thicker
        cols = channels.columns
        for c in cols:
            figure.append_trace(go.Scatter(x=t, y=channels[c], line=dict(color='black', width=(abs(int(c)) - 1) / 2),
                                        showlegend=False),
                             row=1, col=1)


class BollingerBand(Indicator):
    def __init__(self, window_length, time_frame, num_std=2, tp_style='close'):
        super().__init__(window_length, time_frame, tp_style)
        self.num_std = num_std
        self.tp_style = tp_style

        self.ma = MovingAverage(window_length, time_frame, self.tp_style)

        self.name = 'BB_{}_{}_{}'.format(self.window_length, self.num_std, self.tp_style)

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
        col1 = np.array([np.nan]*(self.window_length-1) + list(tp_batch[:-self.window_length+1]))
        tp_matrix = hankel(col1, tp_batch[-self.window_length:])

        self.ma.batch_fit(candle_batch)
        ma = np.array(self.ma.history)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # numpy complains that std has NaN's but this is OK for us.
            # Make a matrix with each row containing (TP_values_in_window - MA_end_window) then get std over each row.
            std = np.nanstd(tp_matrix - ma[:, np.newaxis]*np.ones((1, self.window_length)), axis=1)

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
        figure.append_trace(go.Scatter(x=t, y=bands['UB'], line=dict(color='black', width=0.5), showlegend=False,
                                       fill=None),
                            row=1, col=1)

        # Plot lower band and fill area:
        figure.append_trace(go.Scatter(x=t, y=bands['LB'], line=dict(color='black', width=0.5), showlegend=False,
                                       fill='tonexty', fillcolor='rgba(255,255,255,0.5)'),
                            row=1, col=1)


class MACD():
    def __init__(self, wl_short, wl_long, wl_signal, time_frame, tp_style):
        ema_keys = ['long', 'short', 'signal']
        ema_val = [wl_long, wl_short, wl_signal]
        self.window_length = dict(zip(ema_keys, ema_val))

        self.tp_style = tp_style
        self.time_frame = time_frame
        self.history = pd.DataFrame()

        self.name = 'MACD_{}_{}_{}_{}'.format(wl_short, wl_long, wl_signal, self.tp_style)

        self.ema_l = ExponentialMovingAverage(self.window_length['long'], self.time_frame, self.tp_style)
        self.ema_s = ExponentialMovingAverage(self.window_length['short'], self.time_frame, self.tp_style)
        self.signal = ExponentialMovingAverage(self.window_length['signal'], self.time_frame, 'other')

    def update(self, candle):
        # TODO
        pass

    def batch_fit(self, candle_batch):
        self.ema_l.batch_fit(candle_batch)
        self.ema_s.batch_fit(candle_batch)

        ema_diff = self.ema_s.history - self.ema_l.history
        self.signal.batch_fit(ema_diff)
        histogram = ema_diff - self.signal.history.values.squeeze()

        self.history = pd.concat([ema_diff, self.signal.history, histogram], axis=1)
        self.history.columns = ['EMA_diff', 'signal', 'hist']

    def plot(self, figure):
        ema_diff = self.history['EMA_diff'].values.squeeze()
        signal = self.history['signal'].values.squeeze()
        hist = self.history['hist'].values.squeeze()

        t = self.history.index

        # Plot the ema and signal line:
        figure.append_trace(go.Scatter(x=t, y=ema_diff, line=dict(color='blue', width=0.5), showlegend=False),
                            row=5, col=1)
        figure.append_trace(go.Scatter(x=t, y=signal, line=dict(color='orange', width=1), showlegend=False),
                            row=5, col=1)
        # Plot histogram:
        figure.append_trace(go.Bar(x=t, y=hist),
                            row=5, col=1)


class RSI(Indicator):
    def __init__(self, time_frame, window_length=14, tp_style='close'):
        super().__init__(window_length, time_frame, tp_style)

        self.name = 'RSI_{}_{}'.format(self.window_length, self.tp_style)

        # In order to match TradingViews RSI:
        # TradingView uses Wilder Smoothing, equal to an EMA of length n:
        n = 2*self.window_length - 1

        self.avg_gain = ExponentialMovingAverage(n, self.time_frame, tp_style='other')
        self.avg_loss = ExponentialMovingAverage(n, self.time_frame, tp_style='other')

    def update(self, candle):
        new_tp = super().get_tp(candle, self.tp_style)
        # Calculate all changes inside the window:
        if self.history.empty:
            # If this is the first time updating add the previous candle to avoid NaNs everywhere
            self.memory.append(new_tp)
        change = new_tp - self.memory[-1]
        # Calculate gains and losses (0 if change is not the right sign)
        gain = change*(change > 0)
        loss = abs(change*(change < 0))
        # Update average gain and loss EMA's:
        new_avg_gain = self.avg_gain.update(gain)
        new_avg_loss = self.avg_loss.update(loss)
        # Calculate RSI:
        if new_avg_loss == 0:
            new_rsi = 100
        else:
            relative_strength = new_avg_gain / new_avg_loss
            new_rsi = 100 - 100 / (1 + relative_strength)

        df = pd.DataFrame([new_rsi])
        df.index = pd.DatetimeIndex([candle.name])
        self.history = self.history.append(df)
        self.memory.append(new_tp)

    def batch_fit(self, candle_batch):
        if not self.history.empty:
            self.history = pd.DataFrame()
            self.memory = deque([np.nan] * (self.window_length + 1), maxlen=self.window_length)
            warnings.warn('Old RSI data has been removed! Make sure that this was your intention', UserWarning)

        changes = candle_batch[self.tp_style].diff()
        gains = abs(changes * (changes > 0))
        losses = abs(changes * (changes < 0))

        self.avg_gain.batch_fit(gains)
        self.avg_loss.batch_fit(losses)
        # TODO: These histories to not have a timestamp! Can do so by making a dataframe of gains/losses before batch
        #  fit

        avg_gain = self.avg_gain.history
        avg_loss = self.avg_loss.history

        relative_strength = avg_gain/avg_loss
        rsi = 100 - 100/(1 + relative_strength)

        df = pd.DataFrame(rsi)
        df.index = pd.DatetimeIndex(candle_batch.index)
        self.history = df
        self.memory.extend(candle_batch.tail(self.window_length))

    def plot(self, figure, color='royalblue'):
        rsi = self.history.values.squeeze()
        t = self.history.index

        overbought = np.ones(t.shape) * 80
        oversold = np.ones(t.shape) * 20

        # Plot the overbought line:
        figure.append_trace(go.Scatter(x=t, y=oversold, line=dict(color='black', width=0.5, dash='dash'), showlegend=False,
                                       fill=None),
                            row=5, col=1)
        # Plot oversold line and fill area:
        figure.append_trace(go.Scatter(x=t, y=overbought, line=dict(color='black', width=0.5, dash='dash'),
                                       showlegend=False,
                                       fill='tonexty', fillcolor='rgba(200,200,200,0.4)'),
                            row=5, col=1)
        # Plot the RSI in front of it:
        figure.append_trace(go.Scatter(x=t, y=rsi, line=dict(color=color, width=1), name=self.name),
                            row=5, col=1)


class Stochastic(Indicator):
    def __init__(self, window_length, time_frame):
        super().__init__(window_length, time_frame, tp_style='other')

        self.short_window = 3
        self.k = MovingAverage(self.short_window, time_frame, self.tp_style)  # fast stochastic
        self.d = MovingAverage(self.short_window, time_frame, self.tp_style)  # slow stochastic

        self.name = 'Stoch_{}_{}'.format(self.window_length, self.short_window)

    def update(self):
        # TODO
        pass

    def batch_fit(self, something):
        if not self.history.empty:
            self.history = []
            self.memory = deque([np.nan] * window_length, maxlen=self.window_length)
            warnings.warn('Old Stochastic data has been removed! Make sure that this was your intention.', UserWarning)

        if type(something) == pd.core.frame.DataFrame:
            # If input is candles, do this:
            # Find the true prices (tp) of all candles:
            tp_array = np.array(super().get_tp(something, self.tp_style))
            data = tp_array
            index = pd.DatetimeIndex(something.index)
        elif issubclass(type(something), Indicator):
            # If input is an indicator, do this:
            data = something.history.values.squeeze()
            index = pd.DatetimeIndex(something.history.index)
            if len(data.shape) > 1:
                raise TypeError('Cannot make a Stochastic of {}.'.format(type(something)))
        else:
            raise TypeError('Cannot make a Stochastic of {}'.format(type(something)))

        # Some smart math do to the calculations efficiently:
        col1 = np.array([np.nan] * (self.window_length - 1) + list(data[:-self.window_length + 1]))
        tp_matrix = hankel(col1, data[-self.window_length:])

        period_high = np.max(tp_matrix, axis=1)
        period_low = np.min(tp_matrix, axis=1)
        stochastic = np.divide(data - period_low, period_high - period_low)

        self.k.batch_fit(stochastic)
        self.d.batch_fit(self.k.history)  # this should go wrong now! Should be self.history!

        # Save stochastic history:
        df = pd.DataFrame.from_dict(stochastic)
        df.index = index
        self.history = self.history.append(df)
        # Save last typical prices in memory:
        self.memory.extend(data[-self.window_length:])

    def plot(self, figure):
        k = self.k.history.values.squeeze()*100
        d = self.d.history.values.squeeze()*100
        t = self.history.index

        overbought = np.ones(t.shape)*90
        oversold = np.ones(t.shape)*10

        # Plot the overbought line:
        figure.append_trace(go.Scatter(x=t, y=oversold, line=dict(color='black', width=0.5, dash='dash'),
                                       showlegend=False, fill=None),
                            row=4, col=1)
        # Plot oversold line and fill area:
        figure.append_trace(go.Scatter(x=t, y=overbought, line=dict(color='black', width=0.5, dash='dash'),
                                       showlegend=False,
                                       fill='tonexty', fillcolor='rgba(200,200,200,0.4)'),
                            row=4, col=1)
        # Plot the stochastics:
        figure.append_trace(go.Scatter(x=t, y=k, line=dict(color='blue', width=1), showlegend=False),
                            row=4, col=1)
        figure.append_trace(go.Scatter(x=t, y=d, line=dict(color='orange', width=2), showlegend=False),
                            row=4, col=1)


