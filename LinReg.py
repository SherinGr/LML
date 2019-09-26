# Linear Regression on Boston Dataset.
# Predicting the price of a house

import pandas as pd
import os
import tensorflow as tf

from sklearn import datasets
import itertools

# Import using pandas:
COLS = ['crim', 'zn', 'indus', 'nox', 'rm', 'age',
        'dis', 'tax', 'ptratio', 'medv']

PATH = os.getcwd() + '\\Data\\'
train = pd.read_csv(PATH+'boston_train.csv',
                    skipinitialspace=True, skiprows=1, names=COLS)
test = pd.read_csv(PATH+'boston_test.csv',
                   skipinitialspace=True, skiprows=1, names=COLS)
predict = pd.read_csv(PATH+'boston_predict.csv',
                      skipinitialspace=True, skiprows=1, names=COLS)

# Split the dataset into features and labels:
FEATURES = COLS[0:-2]
LABEL = COLS[-1]

# Make the data structure:
tf_features = [tf.feature_column.numeric_column(k) for k in FEATURES]

# Define the estimator:
estimator = tf.estimator.LinearRegressor(feature_columns=tf_features,
                                         model_dir='train')
# Feeding batches and shuffling in TF:


def get_input_fn(data_set, num_epochs=None, n_batch=128, shuffle=True):
        return tf.estimator.inputs.pandas_input_fn(
                x=pd.DataFrame({col: data_set[col].values for col in FEATURES}), # the data
                y=pd.Series(data_set[LABEL].values),  # the 'labels'
                batch_size=n_batch,
                num_epochs=num_epochs,
                shuffle=shuffle
        )


estimator.train(input_fn=get_input_fn(train, num_epochs=None, n_batch=128, shuffle=False),
                steps=1000)

result = estimator.evaluate(input_fn=get_input_fn(test, num_epochs=1, n_batch=128, shuffle=False))
# Compare dataset characteristics to loss:
loss_score = result['loss']
data_char = train['medv'].describe()

pred = estimator.predict(input_fn=get_input_fn(predict, num_epochs=1, n_batch=128, shuffle=False))

""" Same Solution with NUMPY """


def prep_data(dataFrame):
    pass