# Classify movie reviews positive or negative

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print('Version: ', tf.__version__)
print('Eager mode: ', tf.executing_eagerly())
print('Hub version: ', hub.__version__)

# Import IMDB dataset:
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train, validation), test = tfds.load(
    name='imdb_reviews',
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True
    )

train_examples_batch, train_labels_batch = next(iter(train.batch(10)))
print(train_examples_batch)

