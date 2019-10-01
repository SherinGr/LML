import numpy as np
import tensorflow as tf

# Create array of 2 random numbers:
x_input = np.random.sample((1,2))
print(x_input)

# Create a placeholder for the data:
# NOTE: size mentioned explicitly now
x = tf.placeholder(tf.float32, shape=[1, 2], name="x")

# Create the dataset:
data = tf.data.Dataset.from_tensor_slices(x)

# Create the pipeline:
iterator = data.make_initializable_iterator()
get_next = iterator.get_next()

# Execute the pipeline with our data-array x_input:
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={x: x_input})
    print(sess.run(get_next))

# NOTE: It seems that sess.run only runs one specific node.
