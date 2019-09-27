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
# convert data to numerical values:
train_np = train.values
test_np = test.values
predict_np = predict.values


def prep_data(data_np):
    """Separate labels and features"""
    feat = data_np[:, :-1]
    lab = data_np[:, -1]
    return feat, lab


train_feat, train_lab = prep_data(train_np)
test_feat, test_lab = prep_data(test_np)
predict_feat = predict_np[:, :-1]  # exclude label column

feature_cols = [tf.feature_column.numeric_column('x', shape=train_feat.shape[1:])]
# note train_feat.shape[1] does not yield a tuple! This statement defines that the
# data has a single feature with dimension 9, as is our numpy data. Note that with
# pandas we have a nine-dimensional feature_cols object!

estimator = tf.estimator.LinearRegressor(feature_columns=feature_cols,
                                         model_dir="train1")

# This inputs all features together in the x dictionary, whereas in pandas
# all features are separated in the x dict.
train_input = tf.estimator.inputs.numpy_input_fn(x={'x': train_feat},
                                                 y=train_lab,
                                                 batch_size=128,
                                                 shuffle=False,
                                                 num_epochs=None)

eval_input = tf.estimator.inputs.numpy_input_fn(x={'x':test_feat},
                                                y=test_lab,
                                                shuffle=False,
                                                batch_size=128,
                                                num_epochs=1)

estimator.train(input_fn=train_input, steps=5000)

result2 = estimator.evaluate(input_fn=eval_input, steps=None)

test_input = tf.estimator.inputs.numpy_input_fn(
        x={'x': predict_feat},
        batch_size=128,
        num_epochs=1,
        shuffle=False)

y = estimator.predict(test_input)

# TODO: I do not understand this at all:
predictions = list(p['predictions'] for p in itertools.islice(y, 2))
