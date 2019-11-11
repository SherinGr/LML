# Cartpole using an environment model (NN)

import numpy as np
import cPickle as pickle
import tensorflow as tf

import matplotlib.pyplot as plt
import math
from modelAny import *

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

import gym
env = gym.make('CartPole-v0')

""" Hyper parameters """
h = 8  # number of hidden layer neurons
lr = 1e-2  # learning rate
gamma = 0.99  # reward discount factor
decay_rate = 0.99  # RMSprop leaky sum decay factor

resume = False
model_bs = 3  # batch size when learning from model
real_bs = 3  # batch size when learning from env

d = 4  # input dimensionality

""" Policy network """
tf.reset_default_graph()

observations = tf.placeholder(name='input_x', shape=[None, d], dtype=tf.float32)

# Network weights:
W1 = tf.get_variable(name='W1', shape=[d, h],
                     initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable(name='W2', shape=[h, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
# Network composition:
layer1 = tf.nn.relu(tf.matmul(observations, W1))
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()

input_y = tf.placeholder(name='input_y', shape=[None, 1], dtype=tf.float32)
advantages = tf.placeholder(name='reward_signal', dtype=tf.float32)
W1grad = tf.placeholder(name='batch_grad1', dtype=tf.float32)
W2grad = tf.placeholder(name='batch_grad2', dtype=tf.float32)
batchGrad = [W1grad, W2grad]

loglik = tf.log(input_y*(input_y-probability) + (1-input_y)*(input_y+probability))
adam = tf.train.AdamOptimizer(learning_rate=lr)
loss = -tf.reduce_mean(loglik * advantages)

newGrads = tf.gradient(loss, tvars)
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

""" Model Network """
h_model = 256

input_data = tf.placeholder(name='input_data', shape=[None, 5], dtype=tf.float32)
with tf.variable_scope('rnnlm'):
    # weights and biases
    softmax_w = tf.get_variable('softmax_w', shape=[h_model, 50])
    softmax_b = tf.get_variable('softmax_b', shape=[50])

previous_state = tf.placeholder(name='previous_state', shape=[None, 5], dtype=tf.float32)

W1M = tf.get_variable(name='W1M', shape=[5, h_model],
                      initializer=tf.contrib.layers.xavier_initializer())
B1M = tf.Variable(name='B1M', initial_value=tf.zeros([h_model]))
layer1M = tf.nn.relu(tf.matmul(previous_state, W1M) + B1M)

W2M = tf.get_variable(name='W2M', shape=[h_model, h_model],
                      initializer=tf.contrib.layers.xavier_initializer())
B2M = tf.Variable(name='B2M', initial_value=tf.zeros([h_model]))
layer2M = tf.nn.relu(tf.matmul([layer1M, W2M]) + B2M)

wO = tf.get_variable(name='wO', shape=[h_model, d],
                     initializer=tf.contrib.layers.xavier_initializer())
wR = tf.get_variable(name='wR', shape=[h_model, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
wD = tf.get_variable(name='wD', shape=[h_model, 1],
                     initializer=tf.contrib.layers.xavier_initializer())

bO = tf.Variable(name='bO', initial_value=tf.zeros([4]))
bR = tf.Variable(name='bR', initial_value=tf.zeros([1]))
bD = tf.Variable(name='bD', initial_value=tf.zeros([1]))

predicted_observation = tf.matmul(layer2M, wO, name='predicted_observation') + bO
predicted_reward = tf.matmul(layer2M, wR, name="predicted_reward") + bR
predicted_done = tf.sigmoid(tf.matmul(layer2M, wD, name="predicted_done") + bD)

true_observation = tf.placeholder(name='true_observation', shape=[None, d], dtype=tf.float32)
true_reward = tf.placeholder(name='true_reward', shape=[None, 1], dtype=tf.float32)
true_done = tf.placeholder(name='true_done', shape=[None, 1], dtype=tf.float32)

predicted_state = tf.concat(values=[predicted_observation, predicted_reward, predicted_done], axis=1)

observation_loss = tf.square(true_observation - predicted_observation)
reward_loss = tf.square(true_reward - predicted_reward)
done_loss = tf.mul(predicted_done, true_done) + tf.mul(1-predicted_done, 1 - true_done)

model_loss = tf.reduce_mean(observation_loss + reward_loss + done_loss)

modelAdam = tf.train.AdamOptimizer(learning_rate=lr)
updateModel = modelAdam.minimize(model_loss)

""" Helper functions """
def reset_grad_buffer(grad_buffer):
    for ix, grad in enumerate(grad_buffer)
        grad_buffer[ix] = grad * 0
    return grad_buffer


def discount_rewards(r):
    # Discount the input array of rewards r
    discounted_r = np.zeros_like(r)
    running_r = 0

    for t in reversed(range(0,len(r))):
        running_r = running_r * gamma + r[t]
        discounted_r[t] = running_r
    return discounted_r

def stepModel(sess, xs, action):

