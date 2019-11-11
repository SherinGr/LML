# Solving the cartpole with DRL:

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

gamma = 0.99  # discount rate


def discounted_reward(r):
    """ Turn reward array r into discounted reward array"""

    discounted_array = np.zeros_like(r)
    running_reward = 0
    for t in reversed(range(0, len(r))):
        running_reward = running_reward*gamma + r[t]
        discounted_array[t] = running_reward

    return discounted_array


class Agent:
    def __init__(self, lr, state_dim, action_dim, hidden_layer_size):
        self.state_in = tf.placeholder(name='state', dtype=tf.float32,
                                       shape=[None, state_dim])
        hidden = slim.fully_connected(self.state_in, hidden_layer_size,
                                      activation_fn=tf.nn.relu,
                                      biases_initializer=None)
        self.output = slim.fully_connected(hidden, action_dim,
                                           activation_fn=tf.nn.softmax,
                                           biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # Training procedure:
        self.reward = tf.placeholder(name='reward', dtype=tf.float32,
                                     shape=[None])
        self.action = tf.placeholder(name='action', dtype=tf.int32,
                                     shape=[None])

        self.indices = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indices)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward)

        params = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(params):
            placeholder = tf.placeholder(name=str(idx)+'_holder',
                                         dtype=tf.float32)
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, params)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, params))


tf.reset_default_graph()
n = 4
m = 2
h = 8

agent = Agent(lr=1e-2, state_dim=n, action_dim=m, hidden_layer_size=h)

num_episodes = 5000
max_steps = 999
update_frequency = 5

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    i = 0
    r_episode = []
    length = []

    gradBuffer = sess.run(tf.trainable_variables())

    # initialize gradients
    for idx, grad in enumerate(gradBuffer):
        gradBuffer[idx] = grad * 0

    while i < num_episodes:
        s = env.reset()
        running_reward = 0
        episode_history = []

        for j in range(max_steps):
            outputs = sess.run(agent.output, feed_dict={agent.state_in: [s]})

            action = np.random.choice(outputs[0], p=outputs[0])
            action = np.argmax(outputs == action)

            s_new, r, dead, _ = env.step(action)
            if i > 3000:
                plot = env.render()

            episode_history.append([s, action, r, s_new])
            s = s_new
            running_reward += r
            if dead:
                # Update the network:
                episode_history = np.array(episode_history)
                episode_history[:, 2] = discounted_reward(episode_history[:, 2])

                feed_dict = {agent.reward: episode_history[:, 2],
                             agent.action: episode_history[:, 1],
                             agent.state_in: np.vstack(episode_history[:, 0])}

                gradients = sess.run(agent.gradients, feed_dict=feed_dict)

                for idx, grad in enumerate(gradients):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(agent.gradient_holders, gradBuffer))
                    sess.run(agent.update_batch, feed_dict=feed_dict)

                    for i2, grad in enumerate(gradBuffer):
                        gradBuffer[i2] = grad * 0

                r_episode.append(running_reward)
                length.append(j)
                break

        if i % 100 == 0:
            print(np.mean(r_episode[-100:]))

        i += 1
