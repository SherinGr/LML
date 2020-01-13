# We will webscrape the data of ETH prices from cryptowat.ch
import cryptowatch as cw
import pandas as pd

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dense, Activation, TimeDistributed

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
    # TODO: I think numpy is actually most convenient since we will be using 3D inputs to the LSTM
    candles_4h = pd.DataFrame(data.of_4h, columns=cols)

    # We remove the volume_quote column, since it provides no additional information. We also remove the timestamp since
    # absolute time does not matter, only relative time, we store the times however, they come in handy later.
    close_stamps = candles_4h.close_timestamp
    candles_4h.drop(columns=['close_timestamp', 'volume_quote'])

    # TODO: Split data in training and test set
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
    def __init__(self, data, timesteps, batch_size):
        self.data = data
        self.timesteps = timesteps
        self.batch_size = batch_size

    def generate_batch(self):
        pass


def setup_model(hidden_size, timesteps, input_dim):
    """ Set up the LSTM model using Keras"""
    # Make the LSTM network:
    model = Sequential()
    # 2 LSTM layers:
    model.add(LSTM(hidden_size, input_shape=(timesteps, input_dim), return_sequences=True))  # batch_size is implicit
    # in input shape (?)
    model.add(LSTM(hidden_size, return_sequences=True))

    # The TimeDistributed wrapper applies another dense layer on the output of each time-step:
    model.add(TimeDistributed(Dense(input_dim-1)))  # the output dimension will only be OHLC, not volume
    model.add(Activation('softmax'))  # add softmax activation to the dense layer

    model.compile(loss='categorical_crossentropy', optimizer='adam', metric=['categorical_accuracy'])
    # TODO: Choose which loss function to use specifically for this problem.

    return model


train, test = load_data()

"""This section will set up the hyperparameters:"""
hidden_size = 200   # number of neurons in the LSTM network
num_candles = 10    # number of timesteps for LSTM input sequence
input_dim = train.shape[1]  # input dimension (OHLC & volume)


model = setup_model(hidden_size=hidden_size, timesteps=num_candles, input_dim=input_dim)
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)


# Train the model using batches (otherwise could use .fit):
train_data_generator = BatchGenerator(data=train, timesteps=num_candles)

num_epochs = 40
model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                    validation_data=valid_data_generator.generate(),
                    validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])

