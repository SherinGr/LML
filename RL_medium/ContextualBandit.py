# Medium tutorial 1.5
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


# The contextual bandit means that there are multiple slot machines (i.e. states)
# in which the agent can be. For each machine there is a different optimal lever.
class Environment:
    def __init__(self):
        self.state = 0
        # The bandits:
        self.bandits = np.array([[0.2, 0, -0.0, -5],
                                 [0.1, -5, 1, 0.25],
                                 [-5, 5, 5, 5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_arms = self.bandits.shape[1]

    def get_bandit(self):
        # go to a random bandit
        self.state = np.random.randint(self.num_bandits)
        return self.state

    def pull_arm(self, chosen_arm):
        # pull the chosen arm of the current bandit
        bandit = self.bandits[self.state, chosen_arm]
        x = np.random.randn(1)
        if x > bandit:
            return 1
        else:
            return -1


class Agent:
    def __init__(self, lr, state_dim, action_dim):
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        # one hot encoding of state:
        state_enc = slim.one_hot_encoding(self.state_in, state_dim)
        output = slim.fully_connected(state_enc, action_dim,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_initializer=tf.ones_initializer())

        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        # Training pipeline
        self.reward = tf.placeholder(name='reward', shape=[1], dtype=tf.float32)
        self.action = tf.placeholder(name='action', shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action, [1])

        self.loss = -tf.log(self.responsible_weight)*self.reward

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        self.update = optimizer.minimize(self.loss)


tf.reset_default_graph()

ContextualBandit = Environment()
Agent = Agent(lr=1e-3, state_dim=ContextualBandit.num_bandits, action_dim=ContextualBandit.num_arms)

weights = tf.trainable_variables()[0]  # extract the trainable variables of the graph

num_episodes = 1e4
r_bandits = np.zeros(ContextualBandit.bandits.shape)

e = 0.1  # exploration rate

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < num_episodes:
        i += 1

        bandit = ContextualBandit.get_bandit()

        if np.random.rand(1) < e:
            chosen_arm = np.random.randint(ContextualBandit.num_arms)
        else:
            chosen_arm = sess.run(Agent.chosen_action, feed_dict={Agent.state_in: [bandit]})

        r = ContextualBandit.pull_arm(chosen_arm)
        r_bandits[bandit, chosen_arm] += r

        # Update the network:
        feed_dict = {Agent.reward: [r],
                     Agent.action: [chosen_arm],
                     Agent.state_in: [bandit]}

        _, w_new = sess.run([Agent.update, weights], feed_dict=feed_dict)

        if i % 500 == 0:
            print("Iteration {}".format(i))
            #print(w_new)

    for k in range(ContextualBandit.num_bandits):
        arm = np.argmax(w_new[k]) + 1
        print("The Agent thinks arm {} for bandit {} is the best.".format(arm, k+1))
        if np.argmax(w_new[k]) == np.argmin(ContextualBandit.bandits[k]):
            print("That was right!")
        else:
            print("That was wrong....")

    #w = sess.run(weights)
    #print(w)