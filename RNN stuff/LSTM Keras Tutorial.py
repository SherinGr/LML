#adventuresinmachinelearning.com/keras-lstm-tutorial

# Some NLP stuff using Word2Vec and LSTM's in keras.
# We use the Penn Tree Bank dataset containing text.

# We use a one-hot encoding for the words, can be improved with word2vec!

import os
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed


from vocab_construction import *


# Pre-processing the data:
file_path = os.path.abspath(__file__)
data_path = os.path.join(os.path.dirname(file_path), "simple-examples\data\\")

def load_data():
    # construct data paths:
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # build vocabulary:
    word_to_int_dict = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_int_dict)
    valid_data = file_to_word_ids(valid_path, word_to_int_dict)
    test_data = file_to_word_ids(test_path, word_to_int_dict)

    vocab_size = len(word_to_int_dict)
    reversed_dict = dict(zip(word_to_int_dict.values(), word_to_int_dict.keys()))

    # test what we did:
    print(train_data[:5])
    print(vocab_size)

    print(" ".join(reversed_dict[x] for x in train_data[100:110]))

    return train_data, valid_data, test_data, vocab_size, reversed_dict


train_data, valid_data, test_data, vocab_size, reversed_dictionary = load_data()


class BatchGenerator:
    def __init__(self, data, num_steps, batch_size, vocab_size, skip_step=5):
        self.data = data
        self.num_steps = num_steps  # number of words per sample
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.skip_step = skip_step  # "window slide step size"

        self.current_idx = 0

    def generate(self):
        """ This function is an infinite generator of batches. If you use the keras fit_generator function,
        this function will be called once for each batch."""
        # x is the current sample, y is the prediction i.e. x shifted 1 word forward
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocab))

        while True:  # when does this become false?
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    self.current_idx = 0  # end of data reached

                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                # one-hot encoding for y:
                temp_y = self.data[self.current_idx + 1: self.current_idx + self.num_steps + 1]
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocab_size)

                self.current_idx += self.skip_step
            yield x, y

num_steps = 30
batch_size = 20
skip_steps = num_steps  # why is this a nice choice?

train_data_generator = BatchGenerator(train_data, num_steps, batch_size,vocab_size, skip_steps)
valid_data_generator = BatchGenerator(valid_data, num_steps, batch_size,vocab_size, skip_steps)

# Make the LSTM network:
model = Sequential()
# this is the word2vec layer:
hidden_size = 500
model.add(Embedding(vocab_size, hidden_size, input_length=num_steps))
# 2 LSTM layers:
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))

use_dropout = True
if use_dropout:
    model.add(Dropout(0.5))

# apply another layer of computation to each time sample:
model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

# Train the model using batches (otherwise could use .fit):
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

num_epochs = 40
model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                    validation_data=valid_data_generator.generate(),
                    validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])

# Check what the model has learned:

