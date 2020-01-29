import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
# Import my own libraries:
import indicators as ind
from data_processing import *

# To render interactive plots from plotly in your browser:
pio.renderers.default = "browser"

""" Constants """
time_frame = '15m'
exchange = 'Binance'  # later can be input to load_data as well
pair = 'ETHUSDT'

""" Loading data """
# Load old data and add possible new data to it:
filename, candles = load_data(exchange, pair, time_frame)
# Save all the data (candles only for now):
save_data(filename, candles)

""" Processing, draw indicators """

EMA13 = ind.ExponentialMovingAverage(window_length=13, time_frame=time_frame)
print('Fitting EMA...')
EMA13.batch_fit(candles)

# ATR Channels:
ATRChannels26 = ind.ATRChannels(window_length=26, time_frame=time_frame)
print('Fitting ATR Channels...')
ATRChannels26.batch_fit(candles)

# Bollinger band:
print('Fitting Bollinger Bands...')
BB20 = ind.BollingerBand(window_length=80, time_frame=time_frame, tp_style='hlc3')
BB20.batch_fit(candles)

# RSI:
print('Fitting RSI...')
RSI = ind.RSI(window_length=14, time_frame=time_frame)
RSI.batch_fit(candles)

# Stochastic RSI:
print('Fitting Stochastic RSI...')
SRSI = ind.StochasticRSI(stoch_length=14, time_frame=time_frame)
SRSI.batch_fit(candles)

# MACD:
print('Fitting MACD...')
MACD = ind.MACD(12, 26, 9, time_frame, 'close')
MACD.batch_fit(candles)

""" Making plots using Plotly """
print('\nPlotting...')
fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                    specs=[[{"rowspan": 2}], [None], [{}], [{}], [{}]],
                    )

# Bollinger Bands on the background:
BB20.plot(fig)
# Then Candlesticks:
fig.append_trace(go.Candlestick(x=candles.index,
                                open=candles['open'],
                                high=candles['high'],
                                low=candles['low'],
                                close=candles['close'],
                                showlegend=False),
                 row=1, col=1)

# fig.update_layout(xaxis_rangeslider_visible=False)

# Then EMA's and ATR channels:
EMA13.plot(fig)
ATRChannels26.plot(fig)
# Extra indicators:
SRSI.plot(fig)
# RSI.plot(fig)
MACD.plot(fig)

fig.show()
