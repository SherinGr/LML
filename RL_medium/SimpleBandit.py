import numpy as np
import tensorflow as tf

# Define the four bandits:
bandit_thresholds = [0.2, 0, -0.2, -5]  # normal distribution threshold above which reward is given
num_bandits = len(bandit_thresholds)


def pullBandit(bandit):
    # Pull the lever and get a reward:
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1


# The RL Agent:
tf.reset_default_graph()

# State to action mapping:
weights = tf.Variable(tf.ones([num_bandits]))
choose_bandit = tf.argmax(weights, 0)

# Write training pipeline:
reward = tf.placeholder(name='reward', shape=[1], dtype=tf.float32)
bandit = tf.placeholder(name='bandit', shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, bandit, [1])

loss = -(tf.log(responsible_weight)*reward)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

# Training time!
num_episodes = 1000
r_bandits = np.zeros(num_bandits)

e = 0.1  # exploration rate

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < num_episodes:
        i += 1
        # choose a bandit to pull
        if np.random.rand(1) < e:
            chosen_bandit = np.random.randint(num_bandits)
        else:
            chosen_bandit = sess.run(choose_bandit)
        # get reward
        r = pullBandit(bandit_thresholds[chosen_bandit])

        # update the weights for the bandits:
        _, response, w_new = sess.run([update, responsible_weight, weights],
                                      feed_dict={reward: [r], bandit: [chosen_bandit]})

        r_bandits[chosen_bandit] += r

        # Print updates each 50 episodes:
        if i % 50 == 0:
            print("Current rewards for the {} bandits: {}".format(num_bandits, r_bandits))
            #print(w_new)

    print("Finished learning, the agent things bandit {} is the best.".format(np.argmax(w_new+1)))
    if np.argmax(w_new) == np.argmin(np.array(bandit_thresholds)):
        print("That was right!")
    else:
        print("Well, that was wrong...")

