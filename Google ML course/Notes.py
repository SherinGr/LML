# This file contains witty coding snippets from the google ML course that I want to remember.

""" Pandas """
import pandas as pd
import numpy as np
# Pandas will automatically fill missing values with NaN's

california_housing = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

# Pandas is good in graphing, for example histograms:
# california_housing.hist('housing_median_age')

# Lambda function:
# short function that can be implemented without a function env
population = pd.Series([200, 100, 4000])
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
city_data = pd.DataFrame({'City name': city_names, 'Population': population})

population.apply(lambda val: val < 3000)
# Adding feature:
city_data['Saint named city'] = city_names.apply(lambda name: name.startswith('San'))

# Reindexing:
city_data.reindex([2, 0, 1])
city_data

city_data.reindex([2, 4, 3, 1, 0])
city_data
# here we add 2 NaN rows because indices 3 and 4 do not exist in the df.

city_data.reindex(np.random.permutation(city_data.index))

""" Tensorflow """
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics

# using the housing data from the previous section

# permute to avoid ordering effects affecting SGD:
california_housing = california_housing.reindex(
    np.random.permutation(california_housing.index))
# scale house value feature for normal learning rates:
california_housing["median_house_value"] /= 1000.0

# extract input feature (number of rooms)
feature = california_housing[["total_rooms"]]
# make a tf feature column for it (using default shape):
feature_cols = [tf.feature_column.numeric_column('total_rooms')]

targets = california_housing["median_house_value"]

# Apply linear regression to the data:
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5)
# gradient clipping ensures convergence

linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_cols,
                                                optimizer=optimizer)


# define the input function for the regressor (batches input for each step):
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """"
    returns tuple of (features, labels) for next batch
    """

    # convert pandas to numpy array dict:
    # NOTE: we better use tensorflow's pandas input_fn here!!!!
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size=batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


_ = linear_regressor.train(input_fn=lambda: my_input_fn(feature, targets), steps=100)

prediction_input_fn = lambda: my_input_fn(feature, targets, num_epochs=1, shuffle=False)
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

predictions = np.array([item['predictions'][0] for item in predictions])
mean_squared_error = metrics.mean_squared_error(predictions, targets)

# NOTE: Why use the lambda thing in line 82? Line 84 does not work without this!!!!
# This is because prediction_input_fn should not be the output of my_input_fn but it should also be a function.
(lambda x, y: x + y)(2, 3)  # returns 5
