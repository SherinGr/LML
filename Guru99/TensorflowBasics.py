# TODO: Needs refactoring to create logical workflow
import tensorflow as tf
import numpy as np

# Below is an example of graph computation, what tensorflow is built to do.
# Without actually computing any values we set up a graph using nodes and edges.

x1 = tf.placeholder(tf.float32, name="x_1")
x2 = tf.placeholder(tf.float32, name="x_2")

x3 = tf.multiply(x1, x2, name="x3")

# This is where we will actually "run" the code with values.
# Note that we can use arrays (tensors) as values!!
# TODO: understand why we don't need global_variable_initializer here

with tf.Session() as session:
    result = session.run(x3, feed_dict={x1: [1, 2, 3],
                                        x2: [4, 5, 6]})
    print(result)

""" The four main types of tensors are:
    - tf.get_variable(name, values, dtype, initializer)
    - tf.constant(value, dtype, name="") 
    - tf.placeholder(dtype, shape, name)
    - tf.SparseTensor
    
    names are optional, as is shape apparently.
"""
# Some examples of constants:
decimal = tf.constant(1.234, tf.float32)
string = tf.constant("Hello", tf.string)
vector = tf.constant([1, 2, 3], tf.int16)
boolean = tf.constant([True, True, False], tf.bool)
matrix = tf.constant([[1, 2], [3, 4]], tf.int32)
matrix3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Data type conversion:
x_float = tf.constant(3.1415926, tf.float32)
x_int = tf.cast(x_float, dtype=tf.int32)
x_int += 3  # note that this works!


# Example of running a node:
# We do not need global_variables_initializer() because there are no variables
sess = tf.Session()
print(sess.run(x_int))
# The way you would do it is as follows:
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#   session.run(init)  # run the graph
# or
#   init.run()
#
# note that constants can be ran without this, but not variables.


# Creating variables:
var1 = tf.get_variable("var1", [1, 2], dtype=tf.int32, initializer=tf.zeros_initializer)
# or, initialize using a constant:
var2 = tf.get_variable("var2", dtype=tf.int32, initializer=matrix)

# Placeholders:
# Initialize nodes without actual values, feeding with feed_dict.
placeholder = tf.placeholder(tf.float32, name="placeholder")

# TODO: understand difference constant vs. variable
# constant = tensor, variable = class?
# variable is initialized with a constant

# Running a graph with constants, 2 ways:
# The code
x = tf.constant([2])
y = tf.constant([4])

multiply = tf.multiply(x, y)
power = tf.pow(placeholder, 2)

# 1: Running the graph way 1
sess = tf.Session()
result_1 = sess.run(multiply)
sess.run(tf.global_variables_initializer())
print(sess.run(var2))

print(result_1)
sess.close()

# 2: another option is (no need to close)
with tf.Session() as sess:
    result_2 = multiply.eval()
    data = np.random.rand(1, 10)
    print(sess.run(power, feed_dict={placeholder: data}))
    print(result_2)
