# Apply NN classifier on MNIST digits, using tensorflow

import numpy as np
import tensorflow as tf

np.random.seed(1337)

# from sklearn.datasets import fetch_mldata
# from sklearn.model_selection import train_test_split
#
# mnist = fetch_mldata

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


new_shape = (x_train.shape[0], x_train.shape[1]*x_train.shape[-1])
x_train = x_train.reshape(new_shape)
x_test = x_test.reshape((x_test.shape[0], new_shape[1]))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
x_test_scaled = scaler.fit_transform(x_test.astype(np.float64))

feature_cols = [tf.feature_column.numeric_column('x', shape=x_train_scaled.shape[1:])]

estimator = tf.estimator.DNNClassifier(feature_columns=feature_cols,
                                       hidden_units=[300, 100],
                                       n_classes=10,
                                       model_dir='trainDNN')

train_input = tf.estimator.inputs.numpy_input_fn(x={'x': x_train_scaled},
                                                 y=y_train.astype(np.int32),
                                                 batch_size=50,
                                                 shuffle=False,
                                                 num_epochs=None)

estimator.train(input_fn=train_input, steps=1000)

eval_input = tf.estimator.inputs.numpy_input_fn(x={'x': x_test_scaled},
                                                y=y_test.astype(np.int32),
                                                batch_size=x_test_scaled.shape[0],
                                                shuffle=False,
                                                num_epochs=1)

res = estimator.evaluate(input_fn=eval_input, steps=None)
print(res)

# Adding regularization and dropout:
estimator_improved = tf.estimator.DNNClassifier(
    feature_columns=feature_cols,
    hidden_units=[300, 100],
    dropout=0.3,
    n_classes=10,
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.01,
        l1_regularization_strength=0.01,
        l2_regularization_strength=0.01
    ),
    model_dir='trainDNN2'
    )

estimator_improved.train(input_fn=train_input, steps=1000)
res2 = estimator_improved.evaluate(input_fn=eval_input, steps=None)
print(res2)
