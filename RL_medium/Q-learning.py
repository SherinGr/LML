# Simple Q-learning agent, very basic, using openAI Gym
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

env = gym.make('FrozenLake-v0')

# Initialize Q-table
Q_table = np.zeros([env.observation_space.n, env.action_space.n])
# Learning parameters
lr = .8  # learning rate
g = .95  #forgetting factor/discount rate

num_episodes = 2000

rList = []  # list of rewards at each step
for i in range(num_episodes):
    # Reset environment and get initial observation:
    s = env.reset()
    r_episode = 0
    dead = False
    steps = 0
    while steps < 99:
        steps += 1
        # Greedy policy, with decreasing noise as exploration:
        a = np.argmax(Q_table[s, :] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
        s_new, r, dead, _ = env.step(a)

        # Update Q-table (TD-learning):
        Q_table[s, a] = Q_table[s, a] + lr*(r + g*np.max(Q_table[s_new, :]) - Q_table[s, a])
        r_episode += r
        s = s_new
        if dead:
            break

    rList.append(r_episode)

fig, ax = plt.subplots()
im = ax.imshow(Q_table)

print('Standard RL done, now proceeding with TF')

""" Tensorflow RL """
tf.reset_default_graph()

# Set up FFW network to choose actions:
state = tf.placeholder(name='state',
                       shape=[1, env.observation_space.n],
                       dtype=tf.float32)
W = tf.Variable(tf.random_uniform(shape=[env.observation_space.n, env.action_space.n],
                                  minval=0,
                                  maxval=0.01))

Q_s = tf.matmul(state, W)  # Q-values for all actions in state "s"
predict = tf.argmax(Q_s, 1)

# Model updating pipeline:
nextQ = tf.placeholder(name='nextQ', shape=[1, env.action_space.n], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Q_s))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

# Training the "network" using RL:
init = tf.global_variables_initializer()

g = 0.99  # discount rate
e = 0.1   # exploration rate
num_episodes = 4000

stepsList = []  # steps per episode
rList = []      # total reward per episode

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # reset environment:
        s = env.reset()
        r_episode = 0
        dead = False
        steps = 0
        # Train the Q-network:
        while steps < 99:
            steps += 1
            # one-hot state encoding:
            s_enc = np.identity(env.observation_space.n)[s:s+1]

            # Greedy policy:
            a, allQ = sess.run([predict, Q_s], feed_dict={state: s_enc})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            # Get new state and reward by performing action:
            s_new, r, dead, _ = env.step(a[0])
            s_new_enc = np.identity(env.observation_space.n)[s_new:s_new+1]
            r_episode += r

            Q_s_new = sess.run(Q_s, feed_dict={state: s_new_enc})
            maxQ = np.max(Q_s_new)
            targetQ = allQ
            targetQ[0, a[0]] = r + g*maxQ

            # Train the model:
            _, W_new = sess.run([updateModel, W], feed_dict={state: s_new_enc, nextQ: targetQ})

            s = s_new
            if dead:
                e = 1. / ((steps/50) + 10)  # reduce exploration
                break

        stepsList.append(steps)
        rList.append(r_episode)


plt.plot(rList)
