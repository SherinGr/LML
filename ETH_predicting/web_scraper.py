# We will webscrape the data of ETH prices from cryptowat.ch
import cryptowatch as cw
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

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
    # absolute time does not matter, only relative time:
    candles_4h = candles_4h.drop(columns=['close_timestamp', 'volume_quote'])

    # TODO: Split data in training and test set later on
    return candles_4h


class BatchGenerator:
    """This class generates a batch of data to be used as input for the LSTM network"""
    def __init__(self, data, num_candles, batch_size, input_dim):
        self.data = np.array(data)  # in case we get a pd.DataFrame as input, we first convert it.
        self.num_candles = num_candles
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.normalizer = MinMaxScaler() # TODO: use some scaling that preserves the information correctly

    def normalize_batch(self, batch, targets):
        batch_norm = self.normalizer.fit_transform(batch)
        targets_norm = self.normalizer.transform(targets)
        # TODO: make sure normalization happens over the right dimension
        # TODO: don't forget to scale back the predictions using inverse_transform()
        # TODO: normalization has to be done per sequence, not over the whole batch!
        return batch_norm, targets_norm

    def generate_batch(self):
        """ This function is an infinite generator of batches. If you use the keras fit_generator function,
            this function will be called once for each batch.
        """
        x = np.zeros((self.batch_size, self.num_candles, self.input_dim))
        y = np.zeros((self.batch_size, self.input_dim-1))  # we will predict the next candle, without volume!

        while True:
            for i in range(self.batch_size):
                if (i+1)*self.num_candles+1 > len(self.data):  # if end of data reached, reset
                    i = 0

                x[i, :, :] = self.data[i*self.num_candles:(i+1)*self.num_candles, :]
                y[i, :] = self.data[(i+1)*self.num_candles+1, :-1]  # do not take volume (last entry) into account!
                # TODO: Add normalization of the sequences
            yield x, y


def setup_model(hidden_size, timesteps, n_features):
    """ Set up the LSTM model using Keras"""
    # Make the LSTM network:
    model = Sequential()
    # 2 LSTM layers:
    model.add(LSTM(hidden_size, input_shape=(timesteps, n_features), return_sequences=True))  # batch_size is implicit
    # in input shape (?)
    model.add(LSTM(hidden_size))

    # The TimeDistributed wrapper applies another dense layer on the output of each time-step:
    model.add(Dense(n_features-1))  # the output dimension will only be OHLC, not volume
    model.add(Activation('linear'))                # TODO: choose a useful activation function

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # TODO: Choose which loss function to use specifically for this problem.

    return model

train_data = load_data()

"""This section will set up the hyperparameters:"""
hidden_size = 200   # number of neurons in the LSTM network
num_candles = 10    # number of timesteps for LSTM input sequence
input_dim = train_data.shape[1]  # input dimension (OHLC & volume)
batch_size = 10


model = setup_model(hidden_size=hidden_size, timesteps=num_candles, n_features=input_dim)
checkpointer = ModelCheckpoint(filepath=os.getcwd() + '/model-{epoch:02d}.hdf5', verbose=2)

# Train the model using batches (otherwise could use .fit):
train_data_generator = BatchGenerator(data=train_data, num_candles=num_candles, batch_size=batch_size, input_dim=input_dim)

num_epochs = 500
train_info = model.fit_generator(train_data_generator.generate_batch(), steps_per_epoch=len(train_data)//(
                    batch_size*num_candles),
                    epochs=num_epochs,
                    validation_data=None,  # valid_data_generator.generate(),
                    validation_steps=None,  # len(valid_data)//(batch_size*num_steps),
                    callbacks=[checkpointer])

# Let us plot some candles to see what we are dealing with:
#fig = go.Figure(data=[go.Candlestick(x=datetime.fromtimestamp(close_stamps),
#                                     open=candles_4h.open,
#                                     high=candles_4h.high,
#                                     low=candles_4h.low,
#                                     close=candles_4h.close)])
#fig.show()


# Plot learning curve and other metrics for performance inspection
mse = train_info.history['mse']
plt.plot(mse)
plt.yscale('log')
plt.title('Learning curve (MSE)')
plt.xlabel('Epoch')
plt.ylabel('MSE ($^2)')
plt.grid(True)
plt.show()


# Make a test prediction:
test = np.array(train_data.iloc[0:num_candles])
test = test.reshape((1, num_candles, 5))

prediction = model.predict(test)
target = np.array(train_data.iloc[num_candles+1])
